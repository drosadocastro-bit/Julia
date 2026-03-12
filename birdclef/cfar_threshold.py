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
