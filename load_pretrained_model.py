"""
load_pretrained_model.py — Checkpoint loading utilities for the ViT model.

Handles loading fine-tuned checkpoints (.pth files) and pretrained encoders
for inference, with automatic device detection and weight mapping.
"""

import os
import torch

from models.vit_model import build_vit_model
from utils.config import get_device, VIT_MODEL_NAME, NUM_CLASSES


# ══════════════════════════════════════════════════
#  CHECKPOINT LOADING
# ══════════════════════════════════════════════════

def load_checkpoint(model_path, device=None, model_name=VIT_MODEL_NAME):
    """
    Load a model (ViT or CLAM) from a checkpoint file.
    Dynamically detects the architecture based on state_dict keys.
    """
    if device is None:
        device = get_device()

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Load checkpoint dictionary
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    # Handle DataParallel wrapped checkpoints
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Detect Architecture
    is_vit = any("cls_token" in k or "patch_embed" in k or "head.bias" in k for k in state_dict.keys())

    if is_vit:
        print(f"[LOAD] Detected ViT architecture for {model_path}")
        model = build_vit_model(
            model_name=model_name,
            num_classes=NUM_CLASSES,
            pretrained=False,
        )
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[LOAD] Model: {model_name} | Params: {param_count:,} | Device: {device}")
        return model, device
    else:
        print(f"[LOAD] Detected CLAM architecture for {model_path}")
        from models.model_clam import CLAM_SB
        import torch.nn as nn

        clam_model = CLAM_SB(
            gate=True,
            size_arg="small",
            dropout=True,
            k_sample=8,
            n_classes=2,
            subtyping=False
        )
        clam_model.load_state_dict(state_dict, strict=False)

        class CLAMWrapper(nn.Module):
            def __init__(self, clam):
                super().__init__()
                self.clam = clam
                import torchvision.models as models
                resnet = models.resnet50(pretrained=True)
                self.feature_extractor = nn.Sequential(
                    resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                    resnet.layer1, resnet.layer2, resnet.layer3,
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.feature_extractor.eval()
                
            def forward(self, batch):
                with torch.no_grad():
                    h = self.feature_extractor(batch)
                    h = h.view(h.size(0), -1)
                A_raw, h_512 = self.clam.attention_net(h)
                patch_logits = self.clam.classifiers(h_512)
                return patch_logits

        model = CLAMWrapper(clam_model)
        model = model.to(device)
        model.eval()
        param_count = sum(p.numel() for p in model.clam.parameters())
        print(f"[LOAD] Model: CLAMWrapper (ResNet50+CLAM_SB) | CLAM Params: {param_count:,} | Device: {device}")
        return model, device



def load_pretrained_encoder(device=None, model_name=VIT_MODEL_NAME):
    """
    Load a pretrained ViT (ImageNet weights) for demo / feature extraction.

    This does NOT load a fine-tuned checkpoint. It uses the ImageNet
    pretrained weights directly, which can serve as a baseline or for
    feature extraction in MIL approaches.

    Args:
        device: torch device (auto-detected if None).
        model_name: timm model identifier.

    Returns:
        (model, device) — pretrained model in eval mode.
    """
    if device is None:
        device = get_device()

    model = build_vit_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=True,
    )

    model = model.to(device)
    model.eval()

    print(f"[LOAD] Loaded pretrained {model_name} (ImageNet weights)")
    return model, device


def load_model_auto(model_path=None, device=None, model_name=VIT_MODEL_NAME):
    """
    Automatically load the best available model.

    Priority:
      1. Fine-tuned checkpoint (if model_path exists)
      2. Pretrained ImageNet encoder (fallback)

    Args:
        model_path: optional path to fine-tuned .pth file.
        device: torch device.
        model_name: timm model identifier.

    Returns:
        (model, device, source_description)
    """
    if device is None:
        device = get_device()

    if model_path and os.path.isfile(model_path):
        model, device = load_checkpoint(model_path, device, model_name)
        return model, device, f"Fine-tuned checkpoint: {model_path}"
    else:
        if model_path:
            print(f"[WARNING] Checkpoint not found at {model_path}, using pretrained ImageNet weights")
        model, device = load_pretrained_encoder(device, model_name)
        return model, device, f"Pretrained {model_name} (ImageNet)"
