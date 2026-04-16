"""
vit_model.py — Vision Transformer architecture definition.

Builds a ViT-Base (vit_base_patch16_224) model using the timm library
with a 2-class classification head for Tumor vs Normal classification.
"""

import torch
import torch.nn as nn
import timm

from utils.config import VIT_MODEL_NAME, NUM_CLASSES


# ══════════════════════════════════════════════════
#  ViT MODEL BUILDER
# ══════════════════════════════════════════════════

def build_vit_model(
    model_name=VIT_MODEL_NAME,
    num_classes=NUM_CLASSES,
    pretrained=True,
    drop_rate=0.1,
):
    """
    Create a Vision Transformer model with a custom classification head.

    Uses timm to instantiate a ViT pretrained on ImageNet-1K and replaces
    the final classification head for binary (Tumor/Normal) classification.

    Args:
        model_name: timm model identifier (default: 'vit_base_patch16_224').
        num_classes: number of output classes (default: 2).
        pretrained: whether to load ImageNet pretrained weights.
        drop_rate: dropout rate for the classification head.

    Returns:
        nn.Module — ViT model ready for training or inference.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    print(f"[MODEL] Built {model_name} | pretrained={pretrained} | classes={num_classes}")
    print(f"[MODEL] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[MODEL] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model


def build_vit_feature_extractor(
    model_name=VIT_MODEL_NAME,
    pretrained=True,
):
    """
    Create a ViT model as a feature extractor (no classification head).

    Useful for Multiple Instance Learning (MIL) approaches where patch
    features are aggregated before classification.

    Args:
        model_name: timm model identifier.
        pretrained: whether to load pretrained weights.

    Returns:
        nn.Module — ViT feature extractor (outputs feature vectors).
        int — feature dimension.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,  # removes classification head, returns features
    )

    # Get feature dimension by running a dummy input
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        feat_dim = model(dummy).shape[-1]

    print(f"[MODEL] Feature extractor: {model_name} | dim={feat_dim}")
    return model, feat_dim


def freeze_backbone(model, unfreeze_last_n_blocks=2):
    """
    Freeze all ViT parameters except the last N transformer blocks
    and the classification head.

    Useful for fine-tuning with limited data to prevent overfitting.

    Args:
        model: ViT model from timm.
        unfreeze_last_n_blocks: number of transformer blocks to keep trainable.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classification head
    if hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True

    # Unfreeze last N transformer blocks
    if hasattr(model, "blocks"):
        total_blocks = len(model.blocks)
        for i in range(max(0, total_blocks - unfreeze_last_n_blocks), total_blocks):
            for param in model.blocks[i].parameters():
                param.requires_grad = True

    # Unfreeze norm layer
    if hasattr(model, "norm"):
        for param in model.norm.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Frozen backbone: {trainable:,}/{total:,} params trainable "
          f"(last {unfreeze_last_n_blocks} blocks + head)")


def get_model_info(model):
    """Return a summary dictionary of model statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "trainable_pct": f"{trainable_params / total_params * 100:.1f}%",
    }
