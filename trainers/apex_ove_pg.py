import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip_apex import clip
from clip_apex.simple_tokenizer import SimpleTokenizer as _Tokenizer

# PG Augmentation
from sklearn.utils import check_random_state
from pypolyagamma import PyPolyaGamma
pg = PyPolyaGamma()

_tokenizer = _Tokenizer()

IMAGENET_TEMPLATES = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
]


CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': 'a photo of a {} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}

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
    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.TRAINER.APEX_OVE_PG.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.APEX_OVE_PG.PROMPT_DEPTH_TEXT, "vision_ctx": cfg.TRAINER.APEX_OVE_PG.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.APEX_OVE_PG.N_CTX_TEXT}
    
    
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

def load_eva_clip_to_cpu(cfg, strict=False):
    from open_clip import create_model_from_pretrained, get_tokenizer
    from clip_eva.model import EvaCLIP, CLIPVisionCfg, CLIPTextCfg
    from collections import defaultdict
    
    model, _ = create_model_from_pretrained('hf-hub:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k')
    state_dict = model.state_dict().items()
    new_state_dict = defaultdict()
    
    for n, p in state_dict:
        if 'visual' in n: new_n = n.replace('trunk.', '')
        elif 'text' in n: new_n = n.replace('text.', '')
        else: new_n = n
        new_state_dict[new_n] = p
    
    vision_cfg, text_cfg = CLIPVisionCfg, CLIPTextCfg
    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.TRAINER.APEX_OVE_PG.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.APEX_OVE_PG.PROMPT_DEPTH_TEXT, "vision_ctx": cfg.TRAINER.APEX_OVE_PG.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.APEX_OVE_PG.N_CTX_TEXT}
    model = EvaCLIP(512, vision_cfg, text_cfg, design_details)
    model.load_state_dict(new_state_dict, strict=strict)
    
    return model

def load_coca_clip_to_cpu(cfg, strict=False):
    from open_clip import create_model_and_transforms
    from clip_coca.coca_model import CoCa
    from clip_coca.model import CLIPVisionCfg, CLIPTextCfg
    from collections import defaultdict
    
    model, *_ = create_model_and_transforms('coca_ViT-B-32', pretrained='laion2b_s13b_b90k')
    state_dict = model.state_dict().items()
    new_state_dict = defaultdict()
    
    for n, p in state_dict:
        if 'text_decoder' not in n: new_state_dict[n] = p
    text_cfg, vision_cfg = CLIPTextCfg(), CLIPVisionCfg()
    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.TRAINER.APEX_OVE_PG.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.APEX_OVE_PG.PROMPT_DEPTH_TEXT, "vision_ctx": cfg.TRAINER.APEX_OVE_PG.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.APEX_OVE_PG.N_CTX_TEXT}
    model = CoCa(512, text_cfg, vision_cfg, design_details)
    model.load_state_dict(new_state_dict, strict=strict)
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        from clip_coca.coca_model import CoCa
        if isinstance(clip_model, CoCa):
            self.is_coca = True
            self.clip_model = clip_model
        else:
            self.is_coca = False
            self.transformer = clip_model.transformer # CLIP, EVA
            self.positional_embedding = clip_model.positional_embedding
            self.ln_final = clip_model.ln_final
            self.text_projection = clip_model.text_projection
            self.dtype = clip_model.dtype
            

    def forward(self, prompts, tokenized_prompts):
        if not self.is_coca:
            x = prompts + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
            return x
        else:
            x, _ = self.clip_model.text.forward_transformer(prompts, tokenized_prompts)
            return x
            

class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.APEX_OVE_PG.PROMPT_DEPTH_TEXT >= 1, "In APEX, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        
        if 'coca' in cfg.MODEL.BACKBONE.NAME.lower():
            from clip_coca import clip as clip
            from open_clip import get_tokenizer
            _tokenizer = get_tokenizer('coca_ViT-B-32')
        else:
            from clip import clip as clip
            _tokenizer = _Tokenizer()
        
        n_ctx = cfg.TRAINER.APEX_OVE_PG.N_CTX_TEXT
        ctx_init = cfg.TRAINER.APEX_OVE_PG.CTX_INIT

        try: dtype = clip_model.dtype
        except: dtype = clip_model.visual.conv1.weight.dtype
        try: ctx_dim = clip_model.ln_final.weight.shape[0] # OpenAI CLIP, EVA-CLIP
        except: ctx_dim = clip_model.text.ln_final.weight.shape[0]
        try: vis_dim = clip_model.visual.output_dim # OpenAI CLIP
        except: vis_dim = clip_model.visual.num_classes # EVA-CLIP
        clip_imsize = clip_model.visual.input_resolution # OpenAI CLIP, EVA-CLIP
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 16:
            # Use given words to initialize context vectors
            # use given words to initialize context vectors
            # ctx_init = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                try: embedding = clip_model.token_embedding(prompt).type(dtype)
                except: embedding = clip_model.text.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"APEX design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.APEX_OVE_PG.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix.replace("#", name) for name in classnames]
        # prompts = [prompt_prefix.format(name) for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            try: embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            except: embedding = clip_model.text.token_embedding(tokenized_prompts).type(dtype)

        

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS


        # register tokenized_prompts for adaptive ensemble
        self.register_buffer("trained_tokenized_prompts", tokenized_prompts)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.original_prompts = prompts
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
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        if hasattr(clip_model, "logit_bias"):
            self.logit_bias = clip_model.logit_bias
        else:
            self.logit_bias = None
        try: self.dtype = clip_model.dtype
        except: self.dtype = clip_model.visual.conv1.weight.dtype
        

        # initialize text adapter
        self.VPT_adapter_text_matrix = torch.nn.Parameter(torch.randn(512, 512))
        self.VPT_adapter_text_matrix.requires_grad = True
        # initailize with torch.eye
        self.VPT_adapter_text_matrix.data = torch.eye(512, 512).cuda()

        
        self.VPT_adapter_text_bias = torch.nn.Parameter(torch.randn(1, 512))
        self.VPT_adapter_text_bias.requires_grad = True
        # initailize with torch.eye
        self.VPT_adapter_text_bias.data = torch.zeros(1, 512).cuda()

        if "eva" in cfg.MODEL.BACKBONE.NAME.lower():
            print(f"Loading EVA-CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
            self.clip_model_temp = load_eva_clip_to_cpu(cfg)
        elif "coca" in cfg.MODEL.BACKBONE.NAME.lower():
            print(f"Loading CoCa-CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
            self.clip_model_temp = load_coca_clip_to_cpu(cfg)
        else:
            print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
            self.clip_model_temp = load_clip_to_cpu(cfg)

        # to get the coefficient
        self.alread_init = False



    def init_coeff(self):
        with torch.no_grad():
            trained_tokenized_prompts = self.prompt_learner.trained_tokenized_prompts
            prompts = self.prompt_learner()
            test_tokenized_prompts = self.prompt_learner.tokenized_prompts
            
            # text feartures of base classes 
            trained_features = self.clip_model_temp.encode_text(trained_tokenized_prompts.cuda()).type(self.dtype)

            # text features of test(novel or base) classes
            test_features = self.clip_model_temp.encode_text(test_tokenized_prompts.cuda()).type(self.dtype)


            # Compute the cosine similarity matrix
            cosine_sim_matrix = F.cosine_similarity(trained_features.unsqueeze(1), test_features.unsqueeze(0), dim=-1)


            # If you still want to compute the difference as before (optional)
            # nearset neighbor
            self.diff_nn = 1.0 - cosine_sim_matrix.max(dim=0)[0].reshape(-1, 1)
            

            # average distance 
            self.diff_avg = 1.0 - F.cosine_similarity(trained_features.unsqueeze(1), test_features.unsqueeze(0), dim=-1).mean(dim=0).reshape(-1, 1)

            self.diff_nn = self.diff_nn * (self.diff_nn > 0)

    def forward(self, image):
        if not self.alread_init:
            self.init_coeff()

        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))

        if isinstance(image_features, tuple): image_features = image_features[0]


        if self.prompt_learner.training:
            text_adapted_features = text_features @ self.VPT_adapter_text_matrix.type(self.dtype) + self.VPT_adapter_text_bias.type(self.dtype)
           
        else:
            with torch.no_grad():
                # image features using pretrained clip
                image_original_features = self.clip_model_temp.encode_image(image.cuda()).type(self.dtype)

                # calculate the adapted features
                text_adapted_features = text_features @self.VPT_adapter_text_matrix.type(self.dtype) + self.VPT_adapter_text_bias.type(self.dtype)

                # scaling factor beta
                beta = 4.0

                # compute the adaptive coefficient alpha using the difference
                diff_coeff = self.diff_avg * (self.diff_nn > 0.05)
                alpha = torch.exp(-beta*diff_coeff).type(self.dtype)
            
                # get text ensembled features
                text_adapted_features = alpha * text_adapted_features + (1-alpha) * text_features
                
                # get image ensembled features
                image_features = alpha.mean().reshape(-1,1) * image_features + (1-alpha.mean().reshape(-1,1)) * image_original_features

        # normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_adapted_features = text_adapted_features / text_adapted_features.norm(dim=-1, keepdim=True)
        
        if self.logit_bias is None:
            logits = logit_scale * image_features @ text_adapted_features.t()
        else:
            logits = logit_scale * image_features @ text_adapted_features.t() + self.logit_bias

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
class APEX_OVE_PG(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.APEX_OVE_PG.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        if "eva" in cfg.MODEL.BACKBONE.NAME.lower():
            print(f"Loading EVA-CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
            clip_model = load_eva_clip_to_cpu(cfg)
        elif "coca" in cfg.MODEL.BACKBONE.NAME.lower():
            print(f"Loading CoCa-CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
            clip_model = load_coca_clip_to_cpu(cfg)
        else:
            print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
            clip_model = load_clip_to_cpu(cfg)
            
        if cfg.TRAINER.APEX_OVE_PG.PREC == "fp32" or cfg.TRAINER.APEX_OVE_PG.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.n_cls = self.model.prompt_learner.n_cls
        
        # Polya-Gamma augmentation
        print("Building affine matrix for OVE Softmax")
        if cfg.TRAINER.APEX_OVE_PG.PREC == "fp32":
            self.batch_A = self.batch_affine_matrix(self.n_cls).float()
            self.batch_A_fp16 = self.batch_A.clone().half()
        else:
            self.batch_A = self.batch_affine_matrix(self.n_cls).half()
            self.batch_A_fp16 = self.batch_A.clone()
        if cfg.TRAINER.APEX_OVE_PG.PREC == "fp32" or cfg.TRAINER.APEX_OVE_PG.PREC == "amp":
            self.batch_A = self.batch_A.to(self.device); self.batch_A_fp16 = self.batch_A_fp16.to(self.device)
        
        self.lambda_1 = cfg.TRAINER.APEX_OVE_PG.get("LAMBDA_1", None)
        self.alpha = cfg.TRAINER.APEX_OVE_PG.get("ALPHA", None)
        self.M = self.cfg.TRAINER.APEX_OVE_PG.get("M", None)
        self.eps = 1e-8
        assert self.lambda_1 is not None, "LAMBDA_1 must be given in cfg.TRAINER.APEX_OVE_PG.LAMBDA_1"
        assert self.alpha is not None, "ALPHA must be given in cfg.TRAINER.APEX_OVE_PG.ALPHA"
        assert self.M is not None, "M must be given in cfg.TRAINER.APEX_OVE_PG.M (It indicates the number of gibbs samples) "

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name and "clip_temp" not in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        print("Building zeroshot CLIP for Knowledge Distillation")
        self.freezed_model = ZeroshotCLIP_KD(cfg, classnames)
        self.freezed_model.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

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
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.APEX_OVE_PG.PREC == "amp" else None

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
        
        prec = self.cfg.TRAINER.APEX_OVE_PG.PREC
        if prec == "amp":
            self.optim.zero_grad()
            for _ in range(self.M):
                with autocast():
                    pred = logit = self.model(image, label) # dim : n \times C
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
            model_state = self._models[name].state_dict()
            epoch = checkpoint["epoch"]

            for name_p, param in state_dict.items():
                if 'trained_tokenized_prompts' in name_p:
                    self._models[name].prompt_learner.trained_tokenized_prompts = torch.ones_like(param)

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)