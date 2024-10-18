import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

#################################
# OVE APPROXIMATION FOR SOFTMAX #
#################################
def affine_matrix(y, C, dtype):
    N = y.size(0)

    A = torch.zeros(N * (C - 1), N * C, dtype=dtype)

    row_ind = torch.arange(N * (C - 1))
    cond = torch.ne(
        torch.arange(C).repeat(N), torch.repeat_interleave(y, C)
    )

    A[
        row_ind,
        torch.repeat_interleave(y, C - 1) * N
        + torch.repeat_interleave(torch.arange(N), C - 1),
    ] = 1.0
    A[
        row_ind,
        torch.arange(C).repeat(N)[cond] * N
        + torch.repeat_interleave(torch.arange(N), C - 1),
    ] = -1.0

    return A

def ove_softmax(f, A, axis):
    # f: N x C
    assert f.ndim == 2
    assert axis == -1

    # 로그 공간에서 연산
    logits = torch.einsum("nk,jhk->njh", f, A)
    log_sigmoid = -F.softplus(-logits)  
    sum_log_sigmoid = log_sigmoid.sum(dim=-1)  
    result = torch.exp(sum_log_sigmoid)  
    return result

def ove_log_softmax(f, A, axis):
    # f: N x C
    assert f.ndim == 2
    assert axis == -1

    logits = torch.einsum("nk,jhk->njh", f, A)
    log_sigmoid = -F.softplus(-logits)  
    sum_log_sigmoid = log_sigmoid.sum(dim=-1)  
    result = sum_log_sigmoid
    return result

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP_OVE.N_CTX
        ctx_init = cfg.TRAINER.COCOOP_OVE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        if cfg.TRAINER.COCOOP_OVE.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        
        return logits

#################
# ZEROSHOT CLIP #
#################
CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

class ZeroshotCLIP_KD(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.classnames = classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.requires_grad_(False)
        
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts for ZeroshotCLIP(KD) : {prompts}")
        
        self.prompts = torch.cat([clip.tokenize(p) for p in prompts])
        self.text_features = None
        self.clip_model = clip_model
        
    def set_text_features(self):
        # Determine the device of self.clip_model
        device = next(self.clip_model.parameters()).device
        
        # device
        prompts = self.prompts.to(device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features
    
    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class CoCoOp_OVE(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP_OVE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP_OVE.PREC == "fp32" or cfg.TRAINER.COCOOP_OVE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.n_cls = self.model.prompt_learner.n_cls

        print("Building affine matrix for OVE Softmax")
        self.batch_A = self.batch_affine_matrix(self.n_cls).half()
        if cfg.TRAINER.COCOOP_OVE.PREC == "fp32" or cfg.TRAINER.COCOOP_OVE.PREC == "amp":
            self.batch_A = self.batch_A.to(self.device)
        
        self.lambda_1 = cfg.TRAINER.COCOOP_OVE.get("LAMBDA_1", None)
        self.eps = 1e-8
        assert self.lambda_1 is not None, "LAMBDA_1 must be given in cfg.TRAINER.COCOOP_OVE.LAMBDA_1"

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        print("Building zeroshot CLIP for Knowledge Distillation")
        self.freezed_model = ZeroshotCLIP_KD(cfg, classnames)
        self.freezed_model.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        # Set Device
        self.model.to(self.device)
        self.freezed_model.to(self.device); self.freezed_model.to(self.device)
        self.batch_A = self.batch_A.to(self.device)

        # lambda_1 == 0 means that the freezed_model is not required
        if self.lambda_1 == 0:
            self.freezed_model = None
        else:
            # set text features (prompts -> text features)
            self.freezed_model.set_text_features()

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP_OVE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def batch_affine_matrix(self, n_cls):
        # ove matrix
        batch_A = []
        for cls_idx in range(self.n_cls):
            A = affine_matrix(torch.LongTensor([cls_idx]), n_cls, torch.float32)
            batch_A.append(A)
        batch_A = torch.stack(batch_A)
        return batch_A
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.COCOOP_OVE.PREC
        if prec == "amp":
            with autocast():
                pred = logit = self.model(image)

                # OVE approximation for softmax
                log_prob_ove_sm = ove_log_softmax(logit, self.batch_A, axis=-1)
                nll_loss = F.nll_loss(log_prob_ove_sm, label)

                # regularization
                if self.lambda_1 == 0 or self.freezed_model is None:
                    reg_loss = 0
                else:
                    # knowledge distillation
                    learned_pred = self.freezed_model.model_inference(image); learned_pred = learned_pred.detach()
                    reg_loss = (learned_pred - pred).pow(2).mean()
                    
                # loss
                loss = nll_loss + self.lambda_1 * reg_loss
                
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            # logits
            pred = logits = self.model(image)

            # OVE approximation for softmax
            log_prob_ove_sm = ove_log_softmax(logits, self.batch_A, axis=-1)
            nll_loss = F.nll_loss(log_prob_ove_sm, label)

            # regularization
            if self.lambda_1 == 0 or self.freezed_model is None:
                reg_loss = 0
            else:
                # knowledge distillation
                learned_pred = self.freezed_model.model_inference(image) 
                learned_pred = learned_pred.detach()
                reg_loss = (learned_pred - pred).pow(2).mean()
            
            # loss
            loss = nll_loss + self.lambda_1 * reg_loss

            # backward
            self.model_backward_and_update(loss)
            
        loss_summary = {
            "total_loss": loss.item(),
            "nll_loss": nll_loss.item(),
            "reg_loss": (self.lambda_1 * reg_loss).item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)