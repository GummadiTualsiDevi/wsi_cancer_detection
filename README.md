# WSI Cancer Detection — Vision Transformer for Lymph Node Histopathology

> **Vision Transformer (ViT) based metastatic cancer detection in lymph node whole slide images**

⚠️ **DISCLAIMER**: This system is for **educational and research purposes only**.
It is a decision-support tool and is **NOT** suitable for clinical diagnosis.
Always consult a qualified pathologist.

---

## Project Overview

An end-to-end AI system that analyzes whole slide histopathology images (WSI) of
lymph node biopsies to detect metastatic cancer. The pipeline uses a **Vision Transformer
(ViT-Base)** pretrained on ImageNet and fine-tuned for binary classification
(Tumor vs Normal).

### Key Capabilities

- **WSI Support** — Load `.svs`, `.tif`, `.tiff`, `.ndpi` formats via OpenSlide
- **Standard Image Support** — Also works with `.jpg`, `.png` images
- **Automated Tissue Detection** — Otsu thresholding + morphological segmentation
- **Patch-Level Inference** — Sliding window → batched ViT inference
- **Tumor Heatmap** — Color-mapped probability overlay on the slide
- **Slide-Level Prediction** — Multiple aggregation methods (max, mean, top-K, combined)
- **Suspicious Region Detection** — Coordinates and risk levels for flagged patches
- **Interactive Streamlit UI** — Upload, analyze, visualize, and download results

## Architecture

```
Whole Slide Image (.svs / .tif / .jpg / .png)
       │
       ▼
┌─────────────────────┐
│ OpenSlide / PIL      │  ← Load slide, read metadata, pyramid levels
│ SlideWrapper         │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Tissue Detection     │  ← Grayscale → Otsu → morphological cleanup
│ (tissue_filter.py)   │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Patch Extraction     │  ← 224×224 sliding window, skip background
│ (patch_extractor.py) │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ ViT Batch Inference  │  ← vit_base_patch16_224 (timm), GPU-accelerated
│ (patch_inference.py) │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Slide Prediction     │  ← Aggregate patches → slide cancer probability
│ + Heatmap Generation │  ← Probability grid → JET colormap → overlay
└─────────────────────┘
```

## Project Structure

```
wsi_cancer_detection/
├── models/
│   ├── __init__.py
│   ├── vit_model.py              # ViT architecture (timm)
│   └── load_pretrained_model.py  # Checkpoint loading
├── data/
│   ├── __init__.py
│   ├── patch_extractor.py        # WSI/image patch extraction
│   └── tissue_filter.py          # Background removal
├── inference/
│   ├── __init__.py
│   ├── patch_inference.py        # Batch ViT inference
│   └── slide_prediction.py       # Slide-level aggregation
├── visualization/
│   ├── __init__.py
│   └── heatmap_generator.py      # Heatmap overlays
├── app/
│   ├── __init__.py
│   └── streamlit_app.py          # Web UI
├── utils/
│   ├── __init__.py
│   ├── config.py                 # Constants & transforms
│   └── slide_utils.py            # OpenSlide helpers
├── train.py                      # Training pipeline
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Prerequisites

- **Python 3.9+**
- **(Optional)** NVIDIA GPU with CUDA for faster inference
- **(Optional)** OpenSlide for WSI format support

### 2. Install OpenSlide (for WSI support)

**Windows:**
1. Download from [OpenSlide Windows Binaries](https://openslide.org/download/)
2. Extract and add the `bin/` folder to your system PATH
3. Or use: `pip install openslide-python` (requires OpenSlide C library)

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install openslide-tools python3-openslide
```

**macOS:**
```bash
brew install openslide
```

> **Note:** OpenSlide is optional. Without it, the system still works with
> standard images (JPG, PNG). WSI formats require OpenSlide.

### 3. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Prepare Dataset (for training)

The model expects pre-extracted 224×224 patches in ImageFolder structure:

```
dataset/
├── tumor/      ← tumor patch images (.png / .jpg)
└── normal/     ← normal patch images (.png / .jpg)
```

**Dataset Sources:**
- [PatchCamelyon (PCam)](https://github.com/basveeling/pcam) — 327K patches
- [CAMELYON16](https://camelyon16.grand-challenge.org/) — full WSI slides
- [CAMELYON17](https://camelyon17.grand-challenge.org/) — multi-center

### 6. Train the Model

```bash
# Full fine-tuning
python train.py --data_dir dataset --epochs 10 --batch_size 32

# With backbone freezing (faster, less GPU memory)
python train.py --data_dir dataset --epochs 15 --freeze_backbone --unfreeze_blocks 2

# Quick test with limited samples
python train.py --data_dir dataset --epochs 5 --max_samples 500
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `dataset` | Path to dataset folder |
| `--epochs` | `10` | Training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `1e-4` | Learning rate |
| `--max_samples` | `0` | Max per class (0 = all) |
| `--freeze_backbone` | `false` | Freeze ViT backbone |
| `--unfreeze_blocks` | `2` | Blocks to unfreeze |

Trained model is saved to `models/best_model.pth`.

### 7. Launch Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

> **Note:** If no fine-tuned model is found, the app uses pretrained ImageNet
> weights as a baseline. For accurate cancer detection, train the model first.

### 8. CLI Heatmap Generation

You can also generate heatmaps from the command line (no UI):

```python
from utils.config import get_device
from models.load_pretrained_model import load_checkpoint
from data.patch_extractor import extract_patches
from inference.patch_inference import run_batch_inference
from inference.slide_prediction import build_probability_grid, classify_slide
from visualization.heatmap_generator import generate_slide_heatmap, save_heatmap

# Load model
model, device = load_checkpoint("models/best_model.pth")

# Extract and analyze
patches, grid, mask, slide = extract_patches("slide.svs", patch_size=224, stride=224)
probs, _ = run_batch_inference(model, patches, device)
prob_grid = build_probability_grid(patches, probs, grid)

# Visualize
import numpy as np
from utils.slide_utils import get_slide_thumbnail_np
thumb = get_slide_thumbnail_np("slide.svs")
results = generate_slide_heatmap(thumb, prob_grid)
save_heatmap(results["overlay_bgr"], "outputs/heatmap.png")
```

## Tech Stack

| Technology | Purpose |
|---|---|
| **PyTorch** | Deep learning framework |
| **timm** | Pre-trained Vision Transformer models |
| **OpenSlide** | Whole Slide Image reading |
| **OpenCV** | Image processing & heatmap generation |
| **scikit-image** | Additional image processing |
| **scikit-learn** | Metrics & evaluation |
| **matplotlib / seaborn** | Visualization |
| **Streamlit** | Interactive web application |
| **NumPy / Pandas** | Data manipulation |

## Model Details

| Property | Value |
|---|---|
| Architecture | `vit_base_patch16_224` (Vision Transformer) |
| Pretrained on | ImageNet-1K |
| Fine-tuned for | Binary classification (Tumor vs Normal) |
| Input size | 224 × 224 × 3 |
| Patch size (model) | 16 × 16 |
| Parameters | ~86M |
| Output | 2-class softmax (Normal, Tumor) |

## Outputs

The system produces:
1. **Tumor heatmap overlay** — color-mapped probability overlay on the slide
2. **Slide-level cancer probability** — aggregated prediction score
3. **High-probability patch coordinates** — list with (x, y), probability, risk level
4. **Highlighted tumor regions** — bounding boxes on suspicious areas
5. **Downloadable images** — overlay, heatmap, and annotated versions

---

*B.Tech Final Year Project — WSI Cancer Detection with Vision Transformers*
