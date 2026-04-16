"""
config.py — Central configuration and shared utilities for the WSI Cancer Detection project.

Contains all constants, device detection, image transforms, and filesystem helpers
used across the entire pipeline.
"""

import os
import torch
from torchvision import transforms


# ══════════════════════════════════════════════════
#  PROJECT CONSTANTS
# ══════════════════════════════════════════════════

# --- Image Configuration ---
IMAGE_SIZE = 224                         # ViT input resolution
PATCH_SIZES = [224, 256]                 # Supported patch sizes
DEFAULT_PATCH_SIZE = 224
DEFAULT_STRIDE = 224
DEFAULT_TISSUE_THRESHOLD = 0.3           # Minimum tissue fraction to keep a patch

# --- Model Configuration ---
VIT_MODEL_NAME = "vit_base_patch16_224"  # timm model identifier
NUM_CLASSES = 2                          # Binary: Normal vs Tumor
DEFAULT_BATCH_SIZE = 16                  # Inference batch size

# --- ImageNet Normalization (used by ViT pretrained weights) ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --- Class Labels ---
CLASS_NAMES = ["Normal", "Tumor"]
TUMOR_CLASS_IDX = 1
NORMAL_CLASS_IDX = 0

# --- Prediction Thresholds ---
TUMOR_PROB_THRESHOLD = 0.5               # Patch-level tumor threshold
SLIDE_MAX_PROB_THRESHOLD = 0.5           # Slide-level: max prob to flag tumor
SLIDE_AVG_PROB_THRESHOLD = 0.3           # Slide-level: avg prob co-condition
SLIDE_HIGH_CONFIDENCE_THRESHOLD = 0.7    # High-confidence single-patch detection

# --- Supported File Formats ---
WSI_EXTENSIONS = [".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".vms", ".vmu", ".scn"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
ALL_SUPPORTED_EXTENSIONS = WSI_EXTENSIONS + IMAGE_EXTENSIONS

# --- Heatmap ---
HEATMAP_ALPHA = 0.4                      # Overlay transparency
HEATMAP_COLORMAP = "cv2.COLORMAP_JET"    # OpenCV colormap for heatmap

# --- Directories ---
DEFAULT_MODEL_DIR = "models"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_DATASET_DIR = "dataset"

# --- Disclaimer ---
DISCLAIMER = (
    "⚠️ DISCLAIMER: This system is for educational and research purposes only. "
    "It is a decision-support tool and is NOT intended for clinical diagnosis. "
    "Always consult a qualified pathologist for medical decisions."
)


# ══════════════════════════════════════════════════
#  DEVICE DETECTION
# ══════════════════════════════════════════════════

def get_device():
    """Return the best available torch device (CUDA GPU → CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DEVICE] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[DEVICE] Using CPU")
    return device


# ══════════════════════════════════════════════════
#  IMAGE TRANSFORMS
# ══════════════════════════════════════════════════

def get_train_transforms(image_size=IMAGE_SIZE):
    """Augmentation + normalization transforms for training."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size=IMAGE_SIZE):
    """Normalization-only transforms for validation / inference."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ══════════════════════════════════════════════════
#  FILESYSTEM HELPERS
# ══════════════════════════════════════════════════

def ensure_dir(path):
    """Create a directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def get_file_extension(filepath):
    """Return the lowercase file extension including the dot."""
    return os.path.splitext(filepath)[1].lower()


def is_wsi_file(filepath):
    """Check whether a file is a Whole Slide Image format."""
    return get_file_extension(filepath) in WSI_EXTENSIONS


def is_image_file(filepath):
    """Check whether a file is a standard raster image."""
    return get_file_extension(filepath) in IMAGE_EXTENSIONS
