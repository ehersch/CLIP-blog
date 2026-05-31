"""
CLIP reproduction — supports three model configs:

  small  : T5-small encoder        + ResNet-18    → 256-dim
  large  : DistilBERT encoder      + ResNet-50    → 512-dim
  huge   : MiniLM-L6-v2 encoder    + ViT-S/16     → 512-dim

Pass `config="small"`, `config="large"`, or `config="huge"`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Text encoders
# ---------------------------------------------------------------------------

class T5TextEncoder(nn.Module):
    """T5-small encoder, mean-pooled over non-padding tokens."""
    def __init__(self, embed_dim: int):
        super().__init__()
        from transformers import T5EncoderModel
        self.encoder = T5EncoderModel.from_pretrained("t5-small")
        self.projection = nn.Linear(self.encoder.config.d_model, embed_dim, bias=False)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.projection(pooled)


class DistilBertTextEncoder(nn.Module):
    """DistilBERT encoder, mean-pooled over non-padding tokens."""
    def __init__(self, embed_dim: int):
        super().__init__()
        from transformers import DistilBertModel
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.projection = nn.Linear(self.encoder.config.dim, embed_dim, bias=False)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.projection(pooled)


class MiniLMTextEncoder(nn.Module):
    """all-mpnet-base-v2 — best sentence transformer, fine-tuned for semantic similarity."""
    def __init__(self, embed_dim: int):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        hidden_dim = self.encoder.config.hidden_size  # 768
        self.projection = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.projection(pooled)


# ---------------------------------------------------------------------------
# Image encoders
# ---------------------------------------------------------------------------

class ResNetImageEncoder(nn.Module):
    def __init__(self, variant: str, embed_dim: int):
        """variant: 'resnet18' or 'resnet50'"""
        super().__init__()
        from torchvision.models import (
            resnet18, ResNet18_Weights,
            resnet50, ResNet50_Weights,
        )
        if variant == "resnet18":
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif variant == "resnet50":
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unknown ResNet variant: {variant}")

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Linear(feature_dim, embed_dim, bias=False)

    def forward(self, images):
        return self.projection(self.backbone(images))


class ViTImageEncoder(nn.Module):
    """ViT-S/16 pretrained on ImageNet-21k — CLS token → linear projection."""
    def __init__(self, embed_dim: int):
        super().__init__()
        from transformers import ViTModel
        self.encoder = ViTModel.from_pretrained("WinKawaks/vit-small-patch16-224")
        hidden_dim = self.encoder.config.hidden_size  # 384
        self.projection = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, images):
        # ViT expects pixel_values; uses CLS token (index 0) as image representation
        out = self.encoder(pixel_values=images)
        cls_token = out.last_hidden_state[:, 0, :]  # (B, 384)
        return self.projection(cls_token)


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

CONFIGS = {
    "small": dict(
        text_encoder="t5-small",
        image_encoder="resnet18",
        embed_dim=256,
    ),
    "large": dict(
        text_encoder="distilbert",
        image_encoder="resnet50",
        embed_dim=512,
    ),
    "huge": dict(
        text_encoder="minilm",
        image_encoder="vit-s16",
        embed_dim=512,
    ),
}


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

class CLIP(nn.Module):
    def __init__(self, config: str = "small", init_temperature: float = 0.07):
        super().__init__()
        cfg = CONFIGS[config]
        self.config_name = config
        embed_dim = cfg["embed_dim"]

        # Text encoder
        if cfg["text_encoder"] == "t5-small":
            self.text_encoder = T5TextEncoder(embed_dim)
        elif cfg["text_encoder"] == "distilbert":
            self.text_encoder = DistilBertTextEncoder(embed_dim)
        elif cfg["text_encoder"] == "minilm":
            self.text_encoder = MiniLMTextEncoder(embed_dim)
        else:
            raise ValueError(f"Unknown text encoder: {cfg['text_encoder']}")

        # Image encoder
        if cfg["image_encoder"] == "vit-s16":
            self.image_encoder = ViTImageEncoder(embed_dim)
        else:
            self.image_encoder = ResNetImageEncoder(cfg["image_encoder"], embed_dim)

        # Learnable log-temperature (clipped to prevent logits > 100, paper §2.5)
        self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(max=100)

    def encode_text(self, input_ids, attention_mask):
        return F.normalize(self.text_encoder(input_ids, attention_mask), dim=-1)

    def encode_image(self, images):
        return F.normalize(self.image_encoder(images), dim=-1)

    def forward(self, images, input_ids, attention_mask):
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(input_ids, attention_mask)

        logits = (img_emb @ txt_emb.T) * self.temperature  # (B, B)

        labels = torch.arange(len(images), device=images.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        return {
            "loss": loss,
            "logits": logits,
            "img_emb": img_emb,
            "txt_emb": txt_emb,
            "temperature": self.temperature.item(),
        }


def count_parameters(model: nn.Module) -> dict:
    text_params  = sum(p.numel() for p in model.text_encoder.parameters())
    image_params = sum(p.numel() for p in model.image_encoder.parameters())
    total        = sum(p.numel() for p in model.parameters())
    return {
        "text_encoder_M":  round(text_params  / 1e6, 1),
        "image_encoder_M": round(image_params / 1e6, 1),
        "total_M":         round(total        / 1e6, 1),
    }
