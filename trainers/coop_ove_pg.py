import copy
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# PG Augmentation
from sklearn.utils import check_random_state
from pypolyagamma import PyPolyaGamma
pg = PyPolyaGamma()

_tokenizer = _Tokenizer()

####################################
# OVE PG APPROXIMATION FOR SOFTMAX #
####################################
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

def is_diagonal(matrix):
    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    # Create a mask of non-diagonal elements
    non_diag_mask = ~torch.eye(matrix.shape[0], dtype=bool)
    # Check if all off-diagonal elements are zero
    return torch.all(matrix[non_diag_mask] == 0)

def to_one_hot(labels, num_classes):
    """
    Converts class labels to one-hot encoded vectors.

    Args:
        labels (Tensor): A tensor of class labels of shape (N,) with integer values from 0 to num_classes - 1.
        num_classes (int): The number of classes.

    Returns:
        Tensor: One-hot encoded tensor of shape (N, num_classes).
    """
    # Ensure labels are of integer type
    labels = labels.long()
    
    # Use torch.nn.functional.one_hot to convert labels to one-hot vectors
    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    
    return one_hot

def ove_logit(f, A):
    # 로그 공간에서 연산
    logits = torch.einsum("nk,jhk->njh", f, A)
    return logits

def softmax_ove(logits):
    log_sigmoid = -F.softplus(-logits)  # log(sigmoid(x)) 계산
    sum_log_sigmoid = log_sigmoid.sum(dim=-1)  # 로그 값들의 합
    result = torch.exp(sum_log_sigmoid)  # 지수 함수 적용하여 원래 공간으로 변환
    return result

def log_softmax_ove(logits):
    log_sigmoid = -F.softplus(-logits)  # log(sigmoid(x)) 계산
    sum_log_sigmoid = log_sigmoid.sum(dim=-1)  # 로그 값들의 합
    result = sum_log_sigmoid
    return result

def polya_gamma_sample(b, c, pg=PyPolyaGamma()):
    '''b equals to 1 and c equals to Phi @ beta'''
    assert b.shape == c.shape, "shape mismatch"
    original_dtype = b.dtype
    original_device = b.device
    
    omega = torch.empty_like(b)
    shape = None

    if b.ndim > 1:
        shape = b.shape
        b = b.flatten()
        c = c.flatten()
        omega = omega.flatten()

    # typecast float to double
    b_numpy = b.double().numpy()
    c_numpy = c.double().numpy()
    omega_numpy = omega.double().numpy()
    pg.pgdrawv(b_numpy, c_numpy, omega_numpy)  # pypolyagamma doesn't support torch directly, so we use .numpy()

    if shape is not None:
        b_numpy = b_numpy.reshape(shape)
        c_numpy = c_numpy.reshape(shape)
        omega_numpy = omega_numpy.reshape(shape)
    
    # result
    result = torch.from_numpy(omega_numpy).to(device=original_device, dtype=original_dtype)
    return result

def polya_gamma_mean(b, c, pg=PyPolyaGamma()):
    '''b equals to 1 and c equals to Phi @ beta'''
    assert b.shape == c.shape, "shape mismatch"
    shape = None
    
    # numerical stability
    eps = 1e-8  # small value to prevent division by zero
    min_c = 1e-8  # minimum value for clamping
    max_c = 20  # maximum value for clamping
    max_tanh_input = 10  # threshold to stabilize tanh

    # Ensure c is not too small or too large
    c_clamped = torch.clamp(c, min=min_c, max=max_c)

    # Stabilize tanh input
    c_safe = torch.clamp(0.5 * c_clamped, max=max_tanh_input)
    
    if b.ndim > 1:
        shape = b.shape
        b = b.flatten()
        c = c.flatten()
        c_clamped = c_clamped.flatten()
        c_safe = c_safe.flatten()
        
    # Compute omega with stabilization
    omega = torch.div(b, c_clamped + eps) * torch.tanh(c_safe)
    
    # exceptional
    assert not torch.isnan(omega).any(), "NaN value detected"

    if shape is not None:
        omega = omega.reshape(shape)
    
    # result
    result = omega
    return result

def gaussian_sample(mean, cov, random_state=None):
    std = cov.sqrt()
    result = torch.distributions.Normal(mean, std).sample()
    return result

def gaussian_rsample(mean, cov, random_state=None):
    if cov.ndim > 1:
        if cov.ndim == 2 and cov.shape[-2] == cov.shape[-1]:  # Check if it's a square matrix
            cov = cov.diagonal(dim1=-2, dim2=-1)
            cov = cov[..., None]
        else:
            cov = cov
    else:
        cov = cov[..., None]
    
    result = mean + cov.sqrt() * torch.randn_like(mean)
    return result

def conditional_posterior_weights(logit, kappa, alpha, omega):
    latent_dim = logit.shape[0]
    eye = torch.eye(latent_dim).to(logit.device)

    Sigma_inv = torch.diag(omega) + (alpha * eye)

    # mu
    mu_1 = (alpha * eye) @ logit
    mu_2 = kappa

    mu = torch.linalg.solve(Sigma_inv, mu_1 + mu_2)
    Sigma = torch.linalg.solve(Sigma_inv, eye)
    return mu, Sigma

def conditional_posterior_weights_2(logit, kappa, alpha, omega):
    latent_dim = logit.shape[0]
    eye_1 = torch.eye(latent_dim).to(device=logit.device, dtype=logit.dtype)
    eye_2 = torch.eye(latent_dim).to(device=logit.device, dtype=logit.dtype)
    
    kappa = kappa.to(device=logit.device, dtype=logit.dtype)

    Sigma_inv = torch.diag(omega) + alpha * eye_1

    # mu
    mu_1 = (alpha * eye_2) @ logit
    mu_2 = kappa

    # diagnoal matrix (2D Tensor) and torch.half 
    Sigma_inv_diag = torch.diagonal(Sigma_inv); Sigma_inv_diag = Sigma_inv_diag[..., None] # (latent_dim, 1)
    mu = (mu_1 + mu_2) / Sigma_inv_diag
    Sigma = 1 / Sigma_inv_diag
    
    return mu, Sigma

def conditional_posterior_auxiliary(logits):
    if logits.dtype == torch.float16:
        logits = logits.float()
    
    c = logits
    b = torch.ones_like(logits)
    return b, c

################
# CUSTOME CLIP #
################
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

##############
# CUSTOMCLIP #
##############
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP_OVE_PG.N_CTX
        ctx_init = cfg.TRAINER.COOP_OVE_PG.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
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
            if cfg.TRAINER.COOP_OVE_PG.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
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
        self.class_token_position = cfg.TRAINER.COOP_OVE_PG.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

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

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
    
    def get_features(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

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
class CoOp_OVE_PG(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP_OVE_PG.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP_OVE_PG.PREC == "fp32" or cfg.TRAINER.COOP_OVE_PG.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.n_cls = self.model.prompt_learner.n_cls
        
        print("Building affine matrix for OVE Softmax")
        if cfg.TRAINER.COOP_OVE_PG.PREC == "fp32":
            self.batch_A = self.batch_affine_matrix(self.n_cls).float()
            self.batch_A_fp16 = self.batch_A.clone().half()
        else:
            self.batch_A = self.batch_affine_matrix(self.n_cls).half()
            self.batch_A_fp16 = self.batch_A.clone()
        if cfg.TRAINER.COOP_OVE_PG.PREC == "fp32" or cfg.TRAINER.COOP_OVE_PG.PREC == "amp":
            self.batch_A = self.batch_A.to(self.device); self.batch_A_fp16 = self.batch_A_fp16.to(self.device)
        
        self.lambda_1 = cfg.TRAINER.COOP_OVE_PG.get("LAMBDA_1", None)
        self.alpha = cfg.TRAINER.COOP_OVE_PG.get("ALPHA", None)
        self.M = self.cfg.TRAINER.COOP_OVE_PG.get("M", None)
        self.eps = 1e-8
        assert self.lambda_1 is not None, "LAMBDA_1 must be given in cfg.TRAINER.COOP_OVE_PG.LAMBDA_1"
        assert self.alpha is not None, "ALPHA must be given in cfg.TRAINER.COOP_OVE_PG.ALPHA"
        assert self.M is not None, "M must be given in cfg.TRAINER.COOP_OVE_PG.M (It indicates the number of gibbs samples) "

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        
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

        self.scaler = GradScaler() if cfg.TRAINER.COOP_OVE_PG.PREC == "amp" else None

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
        # feed forward
        image, label = self.parse_batch_train(batch)
        label_one_hot = to_one_hot(label, self.n_cls)
        kappa = label_one_hot - 0.5
        
        prec = self.cfg.TRAINER.COOP_OVE_PG.PREC
        if prec == "amp":
            # zero_grad
            self.optim.zero_grad()
            for _ in range(self.M):
                with autocast():
                    pred = logit = self.model(image) # dim : n \times C
                    learned_pred = self.freezed_model.model_inference(image) # dim : n \times C
                    
                    if pred.dtype == torch.float32:
                        learned_pred = learned_pred.float()
                    
                    # ove logit : (literally, n \times C-1 because the class row corresponds to 0 vector, it can be neglected.)
                    ove_logits = ove_logit(pred, self.batch_A) # dim : n \times C \times C
                    learned_ove_logits = ove_logit(learned_pred, self.batch_A) # dim : n \times C \times C
                    
                    # new logits
                    new_logits = []
                    
                    for cls_idx in range(self.n_cls):
                        # parameters
                        logit = ove_logits[:, cls_idx]; learned_ove_logit = learned_ove_logits[:, cls_idx] # dim : n \times C 
                        kappa_c = kappa[..., cls_idx] # dim : n \times 1 
                        kappa_c = kappa_c.repeat(1, self.n_cls-1) # dim : n \times C (literally, n \times C-1 because the class row corresponds to 0 vector, it can be neglected.)
                        
                        # gibbs sampling: polya-gamma posterior sample
                        with torch.no_grad():
                            b, c = conditional_posterior_auxiliary(learned_ove_logit) # dim : n \times C
                            omega = polya_gamma_mean(b, c, pg=pg) # dim : n \times C
                            
                        # reshape
                        shape = logit.shape
                        logit = logit.reshape(shape[0]*shape[1], -1)
                        kappa_c = kappa_c.reshape(shape[0]*shape[1], -1)
                        omega = omega.flatten()
                        
                        # for posterior f
                        mu, Sigma = conditional_posterior_weights(logit, kappa_c, self.alpha, omega) # dim : n \times C
                        
                        # sampling
                        logit_prime = gaussian_rsample(mu, Sigma) # dim : n \times C
                        
                        # reshape
                        logit_prime = logit_prime.reshape(shape)
                        new_logits.append(logit_prime)
                        
                    # update
                    new_logits = torch.stack(new_logits, dim=-1).transpose(1, 2) # dim : n \times C \times C
                    
                    # softmax by ove_pg / stabilize log probabilities
                    log_prob_ove_pg_sm = log_softmax_ove(new_logits) # dim : n \times C 
                    # log_prob_ove_pg_sm = torch.clamp(log_prob_ove_sm, min=self.eps)
                    
                    nll_loss = F.nll_loss(log_prob_ove_pg_sm, label)
                    
                    # regularization
                    if self.lambda_1 == 0 or self.freezed_model is None:
                        reg_loss = 0
                    else:
                        # knowledge distillation
                        learned_pred = self.freezed_model.model_inference(image); learned_pred = learned_pred.detach()
                        reg_loss = (learned_pred - pred).pow(2).mean()
                        
                    # loss
                    loss = nll_loss + self.lambda_1 * reg_loss
            
                    # backward
                    self.scaler.scale(loss).backward()
            
            # normalize gradient 
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.div_(self.M)
            
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            # zero_grad
            self.model_zero_grad(None)
            
            for _ in range(self.M): 
                pred = logits = self.model(image) # dim : n \times C
                learned_pred = self.freezed_model.model_inference(image) # dim : n \times C
                
                if pred.dtype == torch.float32:
                    learned_pred = learned_pred.float()
                
                # ove logit : (literally, n \times C-1 because the class row corresponds to 0 vector, it can be neglected.)
                ove_logits = ove_logit(pred, self.batch_A) # dim : n \times C \times C
                learned_ove_logits = ove_logit(learned_pred, self.batch_A) # dim : n \times C \times C
                
                # new logits
                new_logits = []
                
                for cls_idx in range(self.n_cls):
                    # parameters
                    logit = ove_logits[:, cls_idx]; learned_ove_logit = learned_ove_logits[:, cls_idx] # dim : n \times C 
                    kappa_c = kappa[..., cls_idx] # dim : n \times 1 
                    kappa_c = kappa_c.repeat(1, self.n_cls-1) # dim : n \times C (literally, n \times C-1 because the class row corresponds to 0 vector, it can be neglected.)
                    
                    # gibbs sampling: polya-gamma posterior sample
                    with torch.no_grad():
                        b, c = conditional_posterior_auxiliary(learned_ove_logit) # dim : n \times C
                        omega = polya_gamma_mean(b, c, pg=pg) # dim : n \times C
                        # omega_2 = polya_gamma_sample(b, c, pg=pg)
                        
                    # reshape
                    shape = logit.shape
                    logit = logit.reshape(shape[0]*shape[1], -1)
                    kappa_c = kappa_c.reshape(shape[0]*shape[1], -1)
                    omega = omega.flatten()
                    
                    # for posterior f
                    # mu, Sigma = conditional_posterior_weights(logit, kappa_c, self.alpha, omega)
                    mu, Sigma = conditional_posterior_weights_2(logit, kappa_c, self.alpha, omega) # dim : n \times C
                    
                    # sampling
                    logit_prime = gaussian_rsample(mu, Sigma) # dim : n \times C
                    
                    # reshape
                    logit_prime = logit_prime.reshape(shape)
                    new_logits.append(logit_prime)
                
                # update
                new_logits = torch.stack(new_logits, dim=-1)
                new_logits = new_logits.transpose(1, 2) # dim : n \times C \times C
                
                # softmax by ove_pg / stabilize log probabilities
                log_prob_ove_pg_sm = log_softmax_ove(new_logits) # dim : n \times C 
                nll_loss = F.nll_loss(log_prob_ove_pg_sm, label)
                
                # regularization
                if self.lambda_1 == 0 or self.freezed_model is None:
                    reg_loss = 0
                else:
                    # knowledge distillation
                    learned_pred = self.freezed_model.model_inference(image); learned_pred = learned_pred.detach()
                    reg_loss = (learned_pred - pred).pow(2).mean()
                    
                # loss
                loss = nll_loss + self.lambda_1 * reg_loss
                
                # backward
                self.model_backward(loss)
                    
            # normalize gradient 
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.div_(self.M)
            
            # update
            self.model_update(None)
            

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

    @torch.no_grad()
    def inter_intra_distance(self, split=None):
        """A inter/intra distance pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        
        num_classes = self.n_cls
        class_dict = {cls: [] for cls in range(num_classes)}

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            image_features = self.model.get_image_features(input)
            
            for i, (im_feature, lb) in enumerate(zip(image_features, label)):
                class_dict[lb.item()].append(im_feature)
            
        # Convert lists in class_dict to tensors for easier manipulation
        for cls in class_dict:
            class_dict[cls] = torch.stack(class_dict[cls])  # Stack feature tensors

        # Calculate intra-class and inter-class distances
        intra_similarity = {}
        inter_similarity = []

        # Average intra-class similarity across classes
        total_similarity = 0
        total_pairs = 0
        
        # Intra-class distances
        for cls, features in class_dict.items():
            if len(features) > 1:  # Skip if only one sample in the class
                # Compute pairwise cosine similarity
                cosine_similarity_matrix = torch.mm(features, features.T)
                
                # Exclude the diagonal (self-similarity)
                mask = torch.eye(cosine_similarity_matrix.size(0), device=cosine_similarity_matrix.device).bool()
                cosine_similarity_matrix = cosine_similarity_matrix.masked_fill(mask, 0)
        
                # Sum
                num_pairs = cosine_similarity_matrix.numel() - cosine_similarity_matrix.size(0)
                class_similarity_sum = cosine_similarity_matrix.sum().item()
                
                # result
                intra_similarity[cls] = cosine_similarity_matrix.mean().item()
                total_similarity += class_similarity_sum # for total average
                total_pairs += num_pairs # for total average

        # Inter-class distances
        class_means = {cls: features.mean(dim=0) for cls, features in class_dict.items()}

        for cls1, mean1 in class_means.items():
            for cls2, mean2 in class_means.items():
                if cls1 < cls2:  # Avoid double-counting
                    # Compute cosine similarity
                    similarity = torch.dot(mean1, mean2).item()
                    inter_similarity.append(similarity)

        # Aggregate intra-class distance
        avg_intra_similarity = total_similarity / total_pairs if total_pairs > 0 else 0
        
        # Aggregate inter-class distance
        avg_inter_similarity = sum(inter_similarity) / len(inter_similarity) if inter_similarity else 0

        # Print results
        print(f"Intra-class distances (average per class): {intra_similarity}")
        print(f"Intra-class distances (total average): {avg_intra_similarity}")
        print(f"Inter-class distance: {inter_similarity}")
        print(f"Average inter-class distance (total average): {avg_inter_similarity}")
        
        return None
