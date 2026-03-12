"""
birdclef.cfar_threshold — CFAR (Constant False Alarm Rate) adaptive thresholding.

Instead of a fixed 0.5 threshold for all species, estimate a per-species
noise floor from the bottom-50% of sigmoid activations and set:

    T_i = μ_noise + k × σ_noise

This adapts to each species' activation distribution:
  - Rare species with low baseline activations get lower thresholds
  - Common/noisy species get higher thresholds
  - k controls false-alarm rate (k=2.0 ≈ 97.7th percentile of noise)

Reference: CFAR is standard in radar signal processing for adaptive
detection in varying noise environments.
"""

from typing import Tuple

import numpy as np


def cfar_adaptive_threshold(
    probs: np.ndarray,
    k: float = 2.0,
    floor: float = 0.05,
    ceiling: float = 0.95,
) -> np.ndarray:
    """
    Compute per-species CFAR thresholds from sigmoid probability matrix.

    Args:
        probs:   (num_windows, num_species) — raw sigmoid output
        k:       multiplier on noise std (higher = fewer false alarms)
        floor:   minimum threshold to prevent triggering on faint noise
        ceiling: maximum threshold to prevent blocking all detections

    Returns:
        (num_species,) threshold vector T
    """
    num_windows, num_species = probs.shape
    thresholds = np.zeros(num_species, dtype=np.float64)

    # Number of windows to treat as "noise" (bottom half)
    n_noise = max(num_windows // 2, 1)

    for i in range(num_species):
        col = probs[:, i]
        sorted_col = np.sort(col)
        noise_samples = sorted_col[:n_noise]

        mu_noise = noise_samples.mean()
        sigma_noise = noise_samples.std()

        t = mu_noise + k * sigma_noise
        thresholds[i] = np.clip(t, floor, ceiling)

    return thresholds


def cfar_adaptive_threshold_with_stats(
    probs: np.ndarray,
    k: float = 2.0,
    floor: float = 0.05,
    ceiling: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute CFAR thresholds and per-species noise sigma values.

    Args:
        probs:   (num_windows, num_species) — raw sigmoid output
        k:       multiplier on noise std
        floor:   minimum threshold clamp
        ceiling: maximum threshold clamp

    Returns:
        thresholds: (num_species,) CFAR threshold vector
        sigma_noise: (num_species,) noise std used per species
    """
    num_windows, num_species = probs.shape
    thresholds = np.zeros(num_species, dtype=np.float64)
    sigma_noise = np.zeros(num_species, dtype=np.float64)

    n_noise = max(num_windows // 2, 1)

    for i in range(num_species):
        col = probs[:, i]
        sorted_col = np.sort(col)
        noise_samples = sorted_col[:n_noise]

        mu_noise = noise_samples.mean()
        sigma = noise_samples.std()
        sigma_noise[i] = sigma

        t = mu_noise + k * sigma
        thresholds[i] = np.clip(t, floor, ceiling)

    return thresholds, sigma_noise


def fixed_threshold(
    probs: np.ndarray,
    t: float = 0.5,
) -> np.ndarray:
    """
    Return a uniform threshold vector (baseline for comparison).

    Args:
        probs: (num_windows, num_species) — only used for shape
        t:     fixed threshold value

    Returns:
        (num_species,) vector filled with t
    """
    return np.full(probs.shape[1], t, dtype=np.float64)


def apply_threshold(
    probs: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Binarize probability matrix using per-species thresholds.

    Args:
        probs:      (num_windows, num_species)
        thresholds: (num_species,)

    Returns:
        (num_windows, num_species) binary prediction matrix
    """
    return (probs >= thresholds[np.newaxis, :]).astype(np.int32)


# ═══════════════════════════════════════════════════════════════
# Phase 2 — Clutter Suppression (MTI-inspired)
# ═══════════════════════════════════════════════════════════════

def estimate_clutter_profile(
    all_window_probs: np.ndarray,
    percentile: float = 25.0,
) -> np.ndarray:
    """
    Estimate ambient clutter profile for a soundscape.

    Analogous to radar clutter map: computes per-species baseline
    activation from the low percentile of all windows.  Species
    genuinely present will activate *above* this ambient floor.

    Args:
        all_window_probs: (num_windows, num_species) — ALL windows
                          from one soundscape / evaluation batch
        percentile:       lower percentile for ambient estimate
                          (25 captures ambient without suppressing
                          genuine detections)

    Returns:
        clutter_map: (num_species,) per-species ambient baseline
    """
    clutter_map = np.percentile(all_window_probs, percentile, axis=0)
    return clutter_map.astype(np.float32)


def suppress_clutter(
    probs: np.ndarray,
    clutter_map: np.ndarray,
    floor: float = 0.0,
) -> np.ndarray:
    """
    Subtract clutter map from activation probabilities (MTI cancellation).

    After subtraction species at ambient level collapse to ~0 while
    species truly present retain a positive residual.

    Apply BEFORE cfar_adaptive_threshold():
        raw → suppress_clutter() → cfar_adaptive_threshold() → apply_threshold()

    Args:
        probs:       (num_windows, num_species) or (num_species,)
        clutter_map: (num_species,) from estimate_clutter_profile()
        floor:       clip floor after subtraction (default 0.0)

    Returns:
        suppressed: same shape as probs, clutter-cancelled activations
    """
    suppressed = probs - clutter_map
    suppressed = np.clip(suppressed, floor, 1.0)
    return suppressed.astype(np.float32)


def cfar_with_clutter_suppression(
    all_window_probs: np.ndarray,
    k: float = 2.0,
    clutter_percentile: float = 25.0,
    floor: float = 0.05,
    ceiling: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full radar-inspired detection pipeline (MTI + CFAR).

    1. Estimate clutter map   (radar: clutter map generation)
    2. Subtract clutter       (radar: MTI cancellation)
    3. CFAR threshold on clean signal
    4. Binary detection decision

    Args:
        all_window_probs:   (num_windows, num_species)
        k:                  CFAR sensitivity constant
        clutter_percentile: ambient estimation percentile
        floor:              CFAR threshold floor
        ceiling:            CFAR threshold ceiling

    Returns:
        detections:  (num_windows, num_species) binary
        thresholds:  (num_species,) per-species CFAR thresholds (post-suppression)
        clutter_map: (num_species,) estimated clutter profile
    """
    # Step 1: build clutter map
    clutter_map = estimate_clutter_profile(all_window_probs, clutter_percentile)

    # Step 2: MTI suppression
    clean_probs = suppress_clutter(all_window_probs, clutter_map)

    # Step 3: CFAR on clean signal
    thresholds = cfar_adaptive_threshold(clean_probs, k=k, floor=floor, ceiling=ceiling)

    # Step 4: detection
    detections = apply_threshold(clean_probs, thresholds)

    return detections, thresholds, clutter_map
