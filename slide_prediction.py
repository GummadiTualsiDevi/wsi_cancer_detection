"""
slide_prediction.py — Slide-level prediction aggregation and classification.

Aggregates patch-level tumor probabilities into a slide-level prediction
using max pooling, mean pooling, and threshold-based classification.
Also identifies suspicious regions with their coordinates.
"""

import numpy as np

from utils.config import (
    TUMOR_PROB_THRESHOLD,
    SLIDE_MAX_PROB_THRESHOLD,
    SLIDE_AVG_PROB_THRESHOLD,
    SLIDE_HIGH_CONFIDENCE_THRESHOLD,
    CLASS_NAMES,
)


# ══════════════════════════════════════════════════
#  PROBABILITY GRID
# ══════════════════════════════════════════════════

def build_probability_grid(patches, probabilities, grid_shape):
    """
    Map patch-level probabilities back to a 2D spatial grid.

    Args:
        patches: list of PatchInfo objects (with .row, .col attributes).
        probabilities: numpy array of tumor probabilities per patch.
        grid_shape: (num_rows, num_cols) of the patch grid.

    Returns:
        prob_grid: 2D numpy array of shape grid_shape, with tumor
                   probabilities at tissue patch positions and 0 elsewhere.
    """
    num_rows, num_cols = grid_shape
    prob_grid = np.zeros((num_rows, num_cols), dtype=np.float32)

    for patch_info, prob in zip(patches, probabilities):
        r, c = patch_info.row, patch_info.col
        if 0 <= r < num_rows and 0 <= c < num_cols:
            prob_grid[r, c] = prob

    return prob_grid


# ══════════════════════════════════════════════════
#  AGGREGATION METHODS
# ══════════════════════════════════════════════════

def aggregate_max_pooling(probabilities):
    """Slide-level score = max probability across all patches."""
    if len(probabilities) == 0:
        return 0.0
    return float(np.max(probabilities))


def aggregate_mean_pooling(probabilities):
    """Slide-level score = mean probability across all patches."""
    if len(probabilities) == 0:
        return 0.0
    return float(np.mean(probabilities))


def aggregate_top_k_mean(probabilities, k=10):
    """Slide-level score = mean of top-K patch probabilities."""
    if len(probabilities) == 0:
        return 0.0
    k = min(k, len(probabilities))
    sorted_probs = np.sort(probabilities)[::-1]
    return float(np.mean(sorted_probs[:k]))


def aggregate_percentile(probabilities, percentile=95):
    """Slide-level score = Nth percentile of patch probabilities."""
    if len(probabilities) == 0:
        return 0.0
    return float(np.percentile(probabilities, percentile))


# ══════════════════════════════════════════════════
#  SLIDE-LEVEL CLASSIFICATION
# ══════════════════════════════════════════════════

def get_slide_cancer_probability(probabilities, method="combined"):
    """
    Compute the overall slide-level cancer probability.

    Args:
        probabilities: numpy array of patch-level tumor probabilities.
        method: aggregation method —
            'max'       — max pooling
            'mean'      — mean pooling
            'top_k'     — mean of top-K patches
            'combined'  — weighted combination of max and top-K

    Returns:
        slide_probability: float between 0 and 1.
    """
    if len(probabilities) == 0:
        return 0.0

    if method == "max":
        return aggregate_max_pooling(probabilities)
    elif method == "mean":
        return aggregate_mean_pooling(probabilities)
    elif method == "top_k":
        return aggregate_top_k_mean(probabilities)
    elif method == "combined":
        max_p = aggregate_max_pooling(probabilities)
        top_k_p = aggregate_top_k_mean(probabilities, k=10)
        # Weighted: emphasize max but stabilize with top-K
        return 0.6 * max_p + 0.4 * top_k_p
    else:
        return aggregate_max_pooling(probabilities)


def classify_slide(probabilities, method="combined"):
    """
    Classify the slide as Tumor Detected or No Tumor Detected.

    Decision logic:
      - If max prob > HIGH_CONFIDENCE threshold → Tumor Detected
      - If max prob > THRESHOLD and avg > AVG_THRESHOLD → Tumor Detected
      - Otherwise → No Tumor Detected

    Args:
        probabilities: numpy array of patch-level tumor probabilities.
        method: aggregation method for slide probability.

    Returns:
        dict with:
            prediction  — "Tumor Detected" or "No Tumor Detected"
            slide_prob  — overall slide cancer probability
            max_prob    — maximum patch probability
            avg_prob    — average patch probability
            suspicious  — count of patches above threshold
            confidence  — confidence level description
    """
    if len(probabilities) == 0:
        return {
            "prediction": "No Tumor Detected",
            "slide_prob": 0.0,
            "max_prob": 0.0,
            "avg_prob": 0.0,
            "suspicious": 0,
            "confidence": "N/A — no tissue patches found",
        }

    max_prob = float(np.max(probabilities))
    avg_prob = float(np.mean(probabilities))
    suspicious = int(np.sum(probabilities > TUMOR_PROB_THRESHOLD))
    slide_prob = get_slide_cancer_probability(probabilities, method)

    # Classification rules
    if max_prob > SLIDE_HIGH_CONFIDENCE_THRESHOLD:
        prediction = "Tumor Detected"
        confidence = "High"
    elif max_prob > SLIDE_MAX_PROB_THRESHOLD and avg_prob > SLIDE_AVG_PROB_THRESHOLD:
        prediction = "Tumor Detected"
        confidence = "Moderate"
    elif slide_prob > TUMOR_PROB_THRESHOLD:
        prediction = "Tumor Detected"
        confidence = "Low"
    else:
        prediction = "No Tumor Detected"
        confidence = "High" if max_prob < 0.2 else "Moderate"

    return {
        "prediction": prediction,
        "slide_prob": slide_prob,
        "max_prob": max_prob,
        "avg_prob": avg_prob,
        "suspicious": suspicious,
        "confidence": confidence,
    }


# ══════════════════════════════════════════════════
#  SUSPICIOUS REGION IDENTIFICATION
# ══════════════════════════════════════════════════

def get_suspicious_regions(patches, probabilities, threshold=TUMOR_PROB_THRESHOLD):
    """
    Identify and return patches with tumor probability above threshold.

    Args:
        patches: list of PatchInfo objects.
        probabilities: numpy array of probabilities.
        threshold: minimum probability to flag as suspicious.

    Returns:
        List of dicts with patch coordinates and probabilities,
        sorted by probability (highest first).
    """
    suspicious = []

    for patch_info, prob in zip(patches, probabilities):
        if prob >= threshold:
            suspicious.append({
                "row": patch_info.row,
                "col": patch_info.col,
                "x": patch_info.x,
                "y": patch_info.y,
                "patch_size": patch_info.patch_size,
                "tumor_probability": float(prob),
                "risk_level": _risk_level(prob),
            })

    # Sort by probability descending
    suspicious.sort(key=lambda r: r["tumor_probability"], reverse=True)
    return suspicious


def _risk_level(prob):
    """Classify a probability into a human-readable risk level."""
    if prob >= 0.9:
        return "Very High"
    elif prob >= 0.7:
        return "High"
    elif prob >= 0.5:
        return "Moderate"
    elif prob >= 0.3:
        return "Low"
    else:
        return "Minimal"


def get_prediction_summary(patches, probabilities, method="combined"):
    """
    Generate a comprehensive prediction summary for the slide.

    Combines classification, suspicious regions, and statistics
    into a single results dictionary.

    Args:
        patches: list of PatchInfo objects.
        probabilities: numpy array of tumor probabilities.
        method: aggregation method.

    Returns:
        dict with all prediction results.
    """
    classification = classify_slide(probabilities, method)
    suspicious_regions = get_suspicious_regions(patches, probabilities)

    return {
        **classification,
        "total_patches": len(patches),
        "suspicious_regions": suspicious_regions,
        "top_suspicious_count": len([r for r in suspicious_regions if r["tumor_probability"] >= 0.7]),
    }
