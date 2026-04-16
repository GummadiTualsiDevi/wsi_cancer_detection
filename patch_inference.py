"""
patch_inference.py — Batch inference on extracted patches using the ViT model.

Processes patches through the pretrained Vision Transformer in mini-batches,
with GPU acceleration and progress reporting for Streamlit integration.
"""

import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.config import get_val_transforms, DEFAULT_BATCH_SIZE, IMAGE_SIZE


# ══════════════════════════════════════════════════
#  BATCH INFERENCE
# ══════════════════════════════════════════════════

def run_batch_inference(
    model,
    patches,
    device,
    batch_size=DEFAULT_BATCH_SIZE,
    image_size=IMAGE_SIZE,
    progress_callback=None,
):
    """
    Run ViT inference on a list of patches in mini-batches.

    Converts each patch to a normalized tensor, batches them, and runs
    forward pass through the model to get tumor probabilities.

    Args:
        model: trained ViT model in eval mode.
        patches: list of PatchInfo objects (each has .image attribute).
        device: torch device.
        batch_size: number of patches per mini-batch.
        image_size: expected model input size.
        progress_callback: optional callable(current, total) for UI updates.

    Returns:
        probabilities: numpy array of tumor probabilities, shape (n_patches,).
        inference_time: wall-clock time for inference in seconds.
    """
    if len(patches) == 0:
        return np.array([]), 0.0

    transform = get_val_transforms(image_size)
    model.eval()

    # ── Prepare all tensors ──
    all_tensors = []
    for patch_info in patches:
        img = patch_info.image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        tensor = transform(img)
        all_tensors.append(tensor)

    # ── Run inference in mini-batches ──
    all_probs = []
    total_batches = (len(all_tensors) + batch_size - 1) // batch_size
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(all_tensors), batch_size):
            batch_tensors = all_tensors[i:i + batch_size]
            batch = torch.stack(batch_tensors).to(device)

            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            tumor_probs = probs[:, 1].cpu().numpy()  # class 1 = Tumor

            all_probs.extend(tumor_probs)

            # Progress reporting
            if progress_callback:
                current_batch = (i // batch_size) + 1
                progress_callback(current_batch, total_batches)

    inference_time = time.time() - start_time

    return np.array(all_probs, dtype=np.float32), inference_time


def predict_single_patch(model, patch_image, device, image_size=IMAGE_SIZE):
    """
    Predict tumor probability for a single patch image.

    Args:
        model: trained ViT model.
        patch_image: PIL Image or numpy array (RGB).
        device: torch device.
        image_size: model input size.

    Returns:
        (predicted_class_name, tumor_probability)
    """
    from utils.config import CLASS_NAMES

    transform = get_val_transforms(image_size)

    if isinstance(patch_image, np.ndarray):
        patch_image = Image.fromarray(patch_image)
    patch_image = patch_image.convert("RGB")

    tensor = transform(patch_image).unsqueeze(0).to(device)  # [1, 3, H, W]

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        tumor_prob = probs[0, 1].item()
        pred_idx = torch.argmax(probs, dim=1).item()

    return CLASS_NAMES[pred_idx], tumor_prob


def predict_batch_from_tensors(model, batch_tensor, device):
    """
    Run inference on pre-transformed batch tensor.

    Args:
        model: trained ViT model.
        batch_tensor: tensor of shape [B, 3, H, W].
        device: torch device.

    Returns:
        tumor_probs: numpy array of shape (B,).
    """
    model.eval()
    with torch.no_grad():
        batch_tensor = batch_tensor.to(device)
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=1)
        tumor_probs = probs[:, 1].cpu().numpy()

    return tumor_probs
