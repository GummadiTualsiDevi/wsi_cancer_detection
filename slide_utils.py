"""
slide_utils.py — OpenSlide helper functions for Whole Slide Image handling.

Provides unified interface to load WSI files (via OpenSlide) and standard
images (via PIL/OpenCV), read metadata, extract regions, and generate thumbnails.

If OpenSlide is not installed, gracefully falls back to PIL-based loading
for standard image formats.
"""

import os
import numpy as np
from PIL import Image

# Try to import OpenSlide; set flag if unavailable
try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("[WARNING] OpenSlide not found. WSI formats (.svs, .tif, .ndpi) will not be supported.")
    print("          Install OpenSlide: https://openslide.org/download/")

from utils.config import is_wsi_file, is_image_file, IMAGE_SIZE


# ══════════════════════════════════════════════════
#  WSI LOADING
# ══════════════════════════════════════════════════

class SlideWrapper:
    """
    Unified wrapper for both OpenSlide WSI files and standard PIL images.
    
    Provides a consistent interface regardless of the source format:
      - dimensions, level_count, level_dimensions
      - read_region(location, level, size)
      - get_thumbnail(size)
      - properties / metadata
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self._is_wsi = is_wsi_file(filepath)
        self._slide = None
        self._pil_image = None

        if self._is_wsi:
            if not OPENSLIDE_AVAILABLE:
                # Try a PIL fallback for .tif/others since many are just image patches
                try:
                    Image.MAX_IMAGE_PIXELS = None
                    self._pil_image = Image.open(filepath).convert("RGB")
                    self._is_wsi = False # Treat as standard image, PIL handles it
                    print(f"[INFO] OpenSlide unavailable. Successfully loaded {self.filename} using PIL fallback.")
                except Exception as e:
                    raise RuntimeError(
                        f"OpenSlide is required to open WSI file '{self.filename}'. "
                        "Please install OpenSlide: https://openslide.org/download/ "
                        f"(PIL fallback failed: {e})"
                    )
            else:
                self._slide = openslide.OpenSlide(filepath)
        else:
            self._pil_image = Image.open(filepath).convert("RGB")

    # ── Properties ──

    @property
    def dimensions(self):
        """Return (width, height) at level 0 (full resolution)."""
        if self._is_wsi:
            return self._slide.dimensions
        return self._pil_image.size

    @property
    def level_count(self):
        """Number of pyramid levels available."""
        if self._is_wsi:
            return self._slide.level_count
        return 1

    @property
    def level_dimensions(self):
        """List of (width, height) for each pyramid level."""
        if self._is_wsi:
            return self._slide.level_dimensions
        return [self._pil_image.size]

    @property
    def level_downsamples(self):
        """Downsample factor for each pyramid level."""
        if self._is_wsi:
            return self._slide.level_downsamples
        return [1.0]

    @property
    def properties(self):
        """Slide metadata dictionary."""
        if self._is_wsi:
            return dict(self._slide.properties)
        w, h = self._pil_image.size
        return {
            "openslide.vendor": "standard-image",
            "openslide.level-count": "1",
            "width": str(w),
            "height": str(h),
        }

    # ── Reading Regions ──

    def read_region(self, location, level, size):
        """
        Read a region from the slide.

        Args:
            location: (x, y) tuple — top-left corner at level 0 coordinates.
            level: pyramid level to read from.
            size: (width, height) of the region to read at the given level.

        Returns:
            PIL.Image in RGB mode.
        """
        if self._is_wsi:
            region = self._slide.read_region(location, level, size)
            return region.convert("RGB")
        else:
            x, y = location
            w, h = size
            return self._pil_image.crop((x, y, x + w, y + h))

    # ── Thumbnail ──

    def get_thumbnail(self, size=(1024, 1024)):
        """
        Get a downsampled thumbnail of the slide.

        Args:
            size: (max_width, max_height) — aspect ratio is preserved.

        Returns:
            PIL.Image (RGB).
        """
        if self._is_wsi:
            return self._slide.get_thumbnail(size)
        else:
            img = self._pil_image.copy()
            img.thumbnail(size, Image.LANCZOS)
            return img

    # ── Metadata Summary ──

    def get_metadata_summary(self):
        """Return a human-readable summary dict of slide metadata."""
        w, h = self.dimensions
        summary = {
            "filename": self.filename,
            "format": "WSI" if self._is_wsi else "Standard Image",
            "dimensions": f"{w:,} × {h:,} px",
            "width": w,
            "height": h,
            "levels": self.level_count,
            "level_dimensions": [
                f"Level {i}: {lw:,} × {lh:,}"
                for i, (lw, lh) in enumerate(self.level_dimensions)
            ],
        }

        if self._is_wsi:
            props = self.properties
            summary["vendor"] = props.get("openslide.vendor", "Unknown")
            summary["magnification"] = props.get(
                "openslide.objective-power",
                props.get("aperio.AppMag", "Unknown")
            )
            summary["mpp_x"] = props.get("openslide.mpp-x", "N/A")
            summary["mpp_y"] = props.get("openslide.mpp-y", "N/A")

        return summary

    # ── Cleanup ──

    def close(self):
        """Close the underlying slide handle."""
        if self._slide is not None:
            self._slide.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


# ══════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════

def load_slide(filepath):
    """
    Open a slide file and return a SlideWrapper.

    Supports both WSI formats (via OpenSlide) and standard images (via PIL).

    Args:
        filepath: path to the slide or image file.

    Returns:
        SlideWrapper instance.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Slide file not found: {filepath}")
    return SlideWrapper(filepath)


def get_slide_thumbnail_np(filepath, max_size=1024):
    """
    Load a slide and return its thumbnail as a NumPy RGB array.

    Args:
        filepath: path to slide file.
        max_size: maximum dimension for the thumbnail.

    Returns:
        numpy array of shape (H, W, 3), dtype uint8.
    """
    with SlideWrapper(filepath) as slide:
        thumb = slide.get_thumbnail((max_size, max_size))
    return np.array(thumb)


def get_best_level_for_downsample(slide_wrapper, target_downsample):
    """
    Find the best pyramid level for a given target downsample factor.

    Args:
        slide_wrapper: SlideWrapper instance.
        target_downsample: desired downsample factor.

    Returns:
        (level_index, actual_downsample) tuple.
    """
    downsamples = slide_wrapper.level_downsamples
    best_level = 0
    best_diff = float("inf")

    for i, ds in enumerate(downsamples):
        diff = abs(ds - target_downsample)
        if diff < best_diff:
            best_diff = diff
            best_level = i

    return best_level, downsamples[best_level]
