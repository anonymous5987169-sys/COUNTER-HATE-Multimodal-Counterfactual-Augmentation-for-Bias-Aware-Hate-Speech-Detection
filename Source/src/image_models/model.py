"""model.py — Image encoders with optional GRL adversarial head.

Supported backbones:
    - CLIP ViT-B/32 vision encoder
    - EfficientNet-B0
"""

import math
import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import CLIPVisionModel

try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except Exception:
        efficientnet_b0 = None
        EfficientNet_B0_Weights = None


# ─── Gradient Reversal Layer (Ganin et al., JMLR 2016) ──────────────────────
class _GradientReversalFn(Function):
    """Autograd function that negates gradients in backward pass."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps _GradientReversalFn as an nn.Module."""

    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0

    def set_lambda(self, val: float):
        self.lambda_ = val

    def forward(self, x):
        return _GradientReversalFn.apply(x, self.lambda_)


def grl_lambda_schedule(epoch: int, total_epochs: int) -> float:
    """DANN schedule: λ = 2/(1+exp(−10p))−1 where p = epoch/total_epochs."""
    p = epoch / max(total_epochs - 1, 1)
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ─── CLIP ViT-B/32 classifier ───────────────────────────────────────────────
class CLIPViTB32Classifier(nn.Module):
    """
    CLIP ViT-B/32 image-encoder classifier with optional adversarial debiasing.

    Parameters
    ----------
    n_groups : int
        Number of identity groups for the adversarial head.
    use_adversarial : bool
        If True, include a gradient-reversal adversarial head.
    dropout : float
        Dropout probability for classification heads.
    freeze_encoder : bool
        If True, CLIP vision encoder weights are frozen.
    """

    def __init__(
        self,
        n_groups: int = 8,
        use_adversarial: bool = True,
        dropout: float = 0.3,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        # Vision-only CLIP encoder.
        try:
            self.backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as exc:
            raise RuntimeError(
                "Failed to load CLIP ViT-B/32 vision encoder. "
                "Ensure internet access or local cache for 'openai/clip-vit-base-patch32'."
            ) from exc

        self.feature_dim = int(self.backbone.config.hidden_size)

        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False

        frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total = sum(1 for p in self.backbone.parameters())
        print(
            f"  Backbone: {frozen}/{total} param tensors frozen "
            f"(freeze_encoder={freeze_encoder})"
        )

        # ── Task head (binary: hate vs non-hate) ─────────────────────────
        self.task_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),
        )

        # ── Adversarial head (predict target_group with gradient reversal)
        self.use_adversarial = use_adversarial
        if use_adversarial:
            self.grl = GradientReversalLayer()
            self.adv_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, n_groups),
            )

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        pooled = outputs.pooler_output  # (B, hidden_size)

        task_logits = self.task_head(pooled).squeeze(-1)

        adv_logits = None
        if self.use_adversarial:
            rev = self.grl(pooled)
            adv_logits = self.adv_head(rev)              # (B, n_groups)

        return task_logits, adv_logits

    # ── helpers ──────────────────────────────────────────────────────────
    def set_grl_lambda(self, val: float):
        if self.use_adversarial:
            self.grl.set_lambda(val)

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return trainable, total

    def get_backbone_parameters(self):
        return self.backbone.parameters()


class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 classifier with optional GRL adversarial head."""

    def __init__(
        self,
        n_groups: int = 8,
        use_adversarial: bool = True,
        dropout: float = 0.3,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        if efficientnet_b0 is None:
            raise RuntimeError(
                "torchvision EfficientNet is unavailable. Install torchvision to use 'efficientnet_b0'."
            )

        weights = None
        if EfficientNet_B0_Weights is not None:
            try:
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            except Exception:
                weights = None

        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.feature_dim = int(backbone.classifier[1].in_features)

        if freeze_encoder:
            for p in self.features.parameters():
                p.requires_grad = False

        frozen = sum(1 for p in self.features.parameters() if not p.requires_grad)
        total = sum(1 for p in self.features.parameters())
        print(
            f"  Backbone: {frozen}/{total} param tensors frozen "
            f"(freeze_encoder={freeze_encoder})"
        )

        self.task_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),
        )

        self.use_adversarial = use_adversarial
        if use_adversarial:
            self.grl = GradientReversalLayer()
            self.adv_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, n_groups),
            )

    def forward(self, x):
        fmap = self.features(x)
        pooled = self.avgpool(fmap).flatten(1)

        task_logits = self.task_head(pooled).squeeze(-1)

        adv_logits = None
        if self.use_adversarial:
            rev = self.grl(pooled)
            adv_logits = self.adv_head(rev)

        return task_logits, adv_logits

    def set_grl_lambda(self, val: float):
        if self.use_adversarial:
            self.grl.set_lambda(val)

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def get_backbone_parameters(self):
        return self.features.parameters()


# ─── Factory ─────────────────────────────────────────────────────────────────
def create_model(
    use_adversarial: bool = True,
    n_groups: int = 8,
    dropout: float = 0.3,
    freeze_blocks: int = 0,
    architecture: str = "clip_vit_b32",
    freeze_encoder: bool = True,
    device=None,
) -> nn.Module:
    """Construct a supported image classifier and print a parameter summary.

    Parameters
    ----------
    device : torch.device or None
        If provided, move the model to this device before returning.
    """
    if architecture == "clip_vit_b32":
        model = CLIPViTB32Classifier(
            n_groups=n_groups,
            use_adversarial=use_adversarial,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
        )
    elif architecture == "efficientnet_b0":
        model = EfficientNetB0Classifier(
            n_groups=n_groups,
            use_adversarial=use_adversarial,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
        )
        if freeze_blocks > 0 and not freeze_encoder:
            n_frozen = 0
            for i, block in enumerate(model.features):
                if i >= freeze_blocks:
                    break
                for p in block.parameters():
                    p.requires_grad = False
                    n_frozen += 1
            print(f"  EfficientNet feature blocks frozen: {freeze_blocks} (param tensors={n_frozen})")
    else:
        raise ValueError(
            f"Unsupported image architecture '{architecture}'. "
            "Supported: 'clip_vit_b32', 'efficientnet_b0'."
        )
    trainable, total = model.count_parameters()
    print(f"  Model: {total:,} total params | {trainable:,} trainable "
          f"({trainable/total*100:.1f} %)")
    if device is not None:
        model = model.to(device)
        print(f"  Model moved to {device}")
    return model


# ─── quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    m = create_model(use_adversarial=True, n_groups=8, architecture="clip_vit_b32")
    x = torch.randn(2, 3, 224, 224)
    task, adv = m(x)
    print(f"task_logits: {task.shape}  adv_logits: {adv.shape}")

    m2 = create_model(use_adversarial=False, architecture="clip_vit_b32")
    task2, adv2 = m2(x)
    print(f"task_logits: {task2.shape}  adv_logits: {adv2}")

    m3 = create_model(use_adversarial=True, n_groups=8, architecture="efficientnet_b0")
    task3, adv3 = m3(x)
    print(f"effnet task_logits: {task3.shape}  adv_logits: {adv3.shape}")
