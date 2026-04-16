"""
heatmap_generator.py — Tumor probability heatmap visualization.

Maps patch-level predictions back to their spatial locations on the slide,
generates color-mapped heatmaps, and creates overlay visualizations with
highlighted tumor regions.
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils.config import HEATMAP_ALPHA, TUMOR_PROB_THRESHOLD, ensure_dir


# ══════════════════════════════════════════════════
#  HEATMAP GENERATION
# ══════════════════════════════════════════════════

def generate_probability_heatmap(prob_grid, target_size, colormap=cv2.COLORMAP_JET):
    """
    Convert a probability grid to a full-size color heatmap.

    Args:
        prob_grid: 2D numpy array (num_rows, num_cols) of probabilities [0,1].
        target_size: (width, height) to resize the heatmap to match slide.
        colormap: OpenCV colormap constant.

    Returns:
        heatmap_color: BGR color-mapped heatmap, same size as slide.
        heatmap_raw: single-channel uint8 heatmap before colormap.
    """
    target_w, target_h = target_size

    # Resize probability grid to match slide dimensions
    prob_resized = cv2.resize(
        prob_grid, (target_w, target_h),
        interpolation=cv2.INTER_LINEAR
    )

    # Clamp values to [0, 1] and convert to uint8
    prob_resized = np.clip(prob_resized, 0.0, 1.0)
    heatmap_raw = (prob_resized * 255).astype(np.uint8)

    # Apply colormap (cold blue → hot red)
    heatmap_color = cv2.applyColorMap(heatmap_raw, colormap)

    return heatmap_color, heatmap_raw


def create_heatmap_overlay(slide_image, heatmap_color, alpha=HEATMAP_ALPHA):
    """
    Alpha-blend the heatmap over the original slide image.

    Args:
        slide_image: original slide as numpy array (RGB or BGR).
        heatmap_color: color-mapped heatmap (BGR), same size as slide.
        alpha: heatmap opacity (0.0 = invisible, 1.0 = fully opaque).

    Returns:
        overlay: blended BGR image.
    """
    # Ensure both images are the same size
    h, w = slide_image.shape[:2]
    if heatmap_color.shape[:2] != (h, w):
        heatmap_color = cv2.resize(heatmap_color, (w, h))

    # Convert RGB to BGR if needed
    if len(slide_image.shape) == 3 and slide_image.shape[2] == 3:
        # Ensure BGR for OpenCV blending
        slide_bgr = slide_image
        if isinstance(slide_image, np.ndarray):
            # Check if likely RGB (from PIL) vs BGR (from OpenCV)
            pass  # We'll handle conversion at the end

    overlay = cv2.addWeighted(slide_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def generate_slide_heatmap(slide_image_rgb, prob_grid, alpha=HEATMAP_ALPHA):
    """
    Complete heatmap pipeline: prob_grid → colormap → overlay on slide.

    This is the main function to call for end-to-end heatmap visualization.

    Args:
        slide_image_rgb: original slide as RGB numpy array.
        prob_grid: 2D numpy array of patch probabilities.
        alpha: overlay transparency.

    Returns:
        dict with:
            overlay_rgb  — RGB overlay image (for display)
            overlay_bgr  — BGR overlay image (for OpenCV/saving)
            heatmap_rgb  — RGB heatmap only (no slide underneath)
            heatmap_bgr  — BGR heatmap only
            prob_resized — probability map resized to slide dimensions
    """
    h, w = slide_image_rgb.shape[:2]
    slide_bgr = cv2.cvtColor(slide_image_rgb, cv2.COLOR_RGB2BGR)

    # Generate colormap heatmap
    heatmap_bgr, heatmap_raw = generate_probability_heatmap(
        prob_grid, (w, h), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Create overlay
    overlay_bgr = cv2.addWeighted(slide_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # Probability map at full resolution
    prob_resized = cv2.resize(prob_grid, (w, h), interpolation=cv2.INTER_LINEAR)

    return {
        "overlay_rgb": overlay_rgb,
        "overlay_bgr": overlay_bgr,
        "heatmap_rgb": heatmap_rgb,
        "heatmap_bgr": heatmap_bgr,
        "prob_resized": prob_resized,
    }


# ══════════════════════════════════════════════════
#  TUMOR REGION HIGHLIGHTING
# ══════════════════════════════════════════════════

def highlight_tumor_regions(
    slide_image_rgb,
    prob_grid,
    patches,
    probabilities,
    threshold=TUMOR_PROB_THRESHOLD,
    stride=224,
    patch_size=224,
    box_color=(255, 0, 0),
    box_thickness=2,
):
    """
    Draw bounding boxes around high-probability tumor regions.

    Args:
        slide_image_rgb: RGB slide image (numpy array).
        prob_grid: 2D probability grid.
        patches: list of PatchInfo objects.
        probabilities: tumor probabilities per patch.
        threshold: minimum probability to highlight.
        stride: patch stride used during extraction.
        patch_size: size of each patch.
        box_color: RGB color tuple for bounding boxes.
        box_thickness: line thickness for boxes.

    Returns:
        annotated_rgb: image with tumor region boxes drawn.
    """
    annotated = slide_image_rgb.copy()
    h, w = annotated.shape[:2]

    for patch_info, prob in zip(patches, probabilities):
        if prob < threshold:
            continue

        x, y = patch_info.x, patch_info.y

        # Scale coordinates to image size
        # (patches may have been extracted at a different level)
        x1 = min(x, w - 1)
        y1 = min(y, h - 1)
        x2 = min(x + patch_size, w)
        y2 = min(y + patch_size, h)

        # Color intensity based on probability
        intensity = min(1.0, prob)
        color = (
            int(255 * intensity),
            int(50 * (1 - intensity)),
            0,
        )

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        # Add probability text
        label = f"{prob:.0%}"
        font_scale = max(0.3, min(0.6, patch_size / 300))
        cv2.putText(
            annotated, label, (x1 + 2, y1 + int(15 * font_scale * 2)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA,
        )

    return annotated


# ══════════════════════════════════════════════════
#  MATPLOTLIB HEATMAP (for higher-quality output)
# ══════════════════════════════════════════════════

def generate_matplotlib_heatmap(prob_grid, figsize=(10, 8), dpi=150):
    """
    Generate a publication-quality heatmap using matplotlib.

    Args:
        prob_grid: 2D numpy array of probabilities.
        figsize: matplotlib figure size.
        dpi: output resolution.

    Returns:
        fig: matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Custom colormap: blue (low) → yellow (mid) → red (high)
    colors_list = ["#2166ac", "#67a9cf", "#fddbc7", "#ef8a62", "#b2182b"]
    cmap = mcolors.LinearSegmentedColormap.from_list("tumor_cmap", colors_list)

    im = ax.imshow(prob_grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tumor Probability", fontsize=12)

    ax.set_title("Cancer Probability Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Patch Column")
    ax.set_ylabel("Patch Row")

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════
#  SAVE UTILITIES
# ══════════════════════════════════════════════════

def save_heatmap(overlay_bgr, output_path):
    """
    Save a heatmap overlay to disk as a PNG file.

    Args:
        overlay_bgr: BGR overlay image (numpy array).
        output_path: path to save the image.
    """
    import os
    ensure_dir(os.path.dirname(output_path) or ".")
    cv2.imwrite(output_path, overlay_bgr)
    print(f"[SAVE] Heatmap overlay → {output_path}")


def encode_image_to_bytes(image_bgr, fmt=".png"):
    """
    Encode a BGR image to bytes (for Streamlit download button).

    Args:
        image_bgr: BGR numpy array.
        fmt: image format ('.png' or '.jpg').

    Returns:
        bytes object.
    """
    _, buffer = cv2.imencode(fmt, image_bgr)
    return buffer.tobytes()
