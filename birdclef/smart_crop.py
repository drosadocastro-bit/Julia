"""
birdclef.smart_crop — CFAR-driven smart cropping for training audio.

Extracts active vocal windows from weakly-labeled BirdCLEF recordings.
Instead of training on random 5s crops (most of which may be silence),
this module detects windows with actual bird vocalizations using
energy-based CFAR thresholding on mel-spectrograms.

Pipeline:
    1. Load full recording
    2. Slide non-overlapping 5s windows
    3. Score each window by mel-spectrogram energy
    4. Apply CFAR adaptive threshold across all windows
    5. Keep only active windows → fewer, higher-quality training samples

Usage:
    python -m birdclef.smart_crop --train-csv data/birdclef-2026/train_metadata.csv \
                                   --audio-dir data/birdclef-2026/train_audio
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from birdclef.config import (
    SAMPLE_RATE, WINDOW_SECONDS, TRAIN_META_CSV, TRAIN_AUDIO_DIR,
    OUTPUT_DIR,
)
from birdclef.features import load_audio_window, audio_to_melspec

logger = logging.getLogger("birdclef.smart_crop")


# ═══════════════════════════════════════════════════════════════════
# Core extraction
# ═══════════════════════════════════════════════════════════════════

def extract_active_windows(
    audio_path: Path,
    species_label: str,
    window_seconds: float = WINDOW_SECONDS,
    sample_rate: int = SAMPLE_RATE,
    cfar_k: float = 1.5,
) -> List[Tuple[np.ndarray, str, float]]:
    """
    Extract windows containing likely bird vocalizations from a recording.

    Steps:
        1. Load the full audio file.
        2. Slide non-overlapping windows of `window_seconds` length.
        3. Compute mel-spectrogram energy for each window.
        4. Apply CFAR adaptive threshold (μ_noise + k × σ_noise)
           across all window energies.
        5. Return only windows whose energy exceeds the threshold.

    Args:
        audio_path:      Path to the .ogg/.wav audio file.
        species_label:   Species code for this recording's weak label.
        window_seconds:  Window length in seconds (default: 5).
        sample_rate:     Target sample rate (default: 32000).
        cfar_k:          CFAR multiplier — higher = stricter filtering.

    Returns:
        List of (audio_chunk, label, confidence) tuples where:
            audio_chunk: 1-D float32 array of shape (sample_rate * window_seconds,)
            label:       species_label string
            confidence:  energy score for this window (higher = more active)
    """
    import librosa

    # Step 1: Load full audio
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        logger.warning("Failed to load %s: %s", audio_path, e)
        return []

    y = y.astype(np.float32)
    samples_per_window = int(sample_rate * window_seconds)

    if len(y) < samples_per_window:
        # File shorter than one window — pad and return as single candidate
        y_padded = np.pad(y, (0, samples_per_window - len(y)), mode="constant")
        mel = audio_to_melspec(y_padded, sr=sample_rate)
        energy = float(np.mean(mel))
        return [(y_padded, species_label, energy)]

    # Step 2: Slide non-overlapping windows
    num_windows = len(y) // samples_per_window
    chunks = []
    energies = []

    for i in range(num_windows):
        start = i * samples_per_window
        end = start + samples_per_window
        chunk = y[start:end]

        # Step 3: Compute mel-spectrogram energy per window
        mel = audio_to_melspec(chunk, sr=sample_rate)
        energy = float(np.mean(mel))

        chunks.append(chunk)
        energies.append(energy)

    energies = np.array(energies, dtype=np.float64)

    # Step 4: CFAR adaptive threshold
    # Use bottom half of energy values as noise estimate
    n_noise = max(len(energies) // 2, 1)
    sorted_energies = np.sort(energies)
    noise_samples = sorted_energies[:n_noise]

    mu_noise = noise_samples.mean()
    sigma_noise = noise_samples.std()
    threshold = mu_noise + cfar_k * sigma_noise

    # Step 5: Keep windows above threshold
    active_windows = []
    for chunk, energy in zip(chunks, energies):
        if energy > threshold:
            active_windows.append((chunk, species_label, energy))

    # Safety: if CFAR rejected everything, keep the single highest-energy window
    # to avoid dropping entire recordings
    if not active_windows and len(chunks) > 0:
        best_idx = int(np.argmax(energies))
        active_windows.append(
            (chunks[best_idx], species_label, energies[best_idx])
        )

    return active_windows


# ═══════════════════════════════════════════════════════════════════
# Dataset builder
# ═══════════════════════════════════════════════════════════════════

def build_smart_crop_dataset(
    train_csv: Path,
    audio_dir: Path,
    output_path: Path = None,
    cfar_k: float = 1.5,
) -> pd.DataFrame:
    """
    Build a smart-crop manifest from the full training set.

    Steps:
        1. Read train_metadata.csv.
        2. For each recording, run extract_active_windows.
        3. Collect results into a manifest DataFrame.
        4. Log original vs smart-crop window counts.
        5. Save manifest to smart_crop_manifest.csv.

    Args:
        train_csv:   Path to train_metadata.csv.
        audio_dir:   Path to train_audio/ directory.
        output_path: Where to save the manifest CSV. Defaults to OUTPUT_DIR.
        cfar_k:      CFAR strictness multiplier.

    Returns:
        DataFrame with columns:
            filename, species, window_index, offset_seconds, confidence
    """
    meta = pd.read_csv(train_csv)

    if output_path is None:
        output_path = OUTPUT_DIR / "smart_crop_manifest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total_naive_windows = 0
    total_smart_windows = 0

    for idx, record in meta.iterrows():
        filename = record["filename"]
        species = record["primary_label"]
        filepath = Path(audio_dir) / filename

        if not filepath.exists():
            logger.warning("Audio file not found: %s", filepath)
            continue

        active = extract_active_windows(
            filepath, species, cfar_k=cfar_k,
        )

        # Count naive windows (what random-crop would produce)
        import librosa
        try:
            file_duration = librosa.get_duration(path=filepath)
        except Exception:
            file_duration = WINDOW_SECONDS
        naive_count = max(1, int(file_duration // WINDOW_SECONDS))
        total_naive_windows += naive_count
        total_smart_windows += len(active)

        for win_idx, (chunk, label, confidence) in enumerate(active):
            rows.append({
                "filename": filename,
                "species": label,
                "window_index": win_idx,
                "offset_seconds": win_idx * WINDOW_SECONDS,
                "confidence": round(confidence, 6),
            })

        if (idx + 1) % 500 == 0:
            logger.info(
                "Processed %d / %d files  (smart: %d / naive: %d windows)",
                idx + 1, len(meta), total_smart_windows, total_naive_windows,
            )

    manifest = pd.DataFrame(rows)

    # Log compression ratio
    if total_naive_windows > 0:
        ratio = total_smart_windows / total_naive_windows
        logger.info(
            "Smart crop: %d windows from %d naive (%.1f%% kept)",
            total_smart_windows, total_naive_windows, ratio * 100,
        )
    else:
        logger.info("No windows processed (empty dataset?).")

    manifest.to_csv(output_path, index=False)
    logger.info("Manifest saved to %s", output_path)

    return manifest


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build smart-crop training manifest via CFAR energy detection."
    )
    parser.add_argument(
        "--train-csv", type=Path, default=TRAIN_META_CSV,
        help="Path to train_metadata.csv",
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=TRAIN_AUDIO_DIR,
        help="Path to train_audio/ directory",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output manifest CSV path",
    )
    parser.add_argument(
        "--cfar-k", type=float, default=1.5,
        help="CFAR threshold multiplier (higher = stricter)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s"
    )

    build_smart_crop_dataset(
        train_csv=args.train_csv,
        audio_dir=args.audio_dir,
        output_path=args.output,
        cfar_k=args.cfar_k,
    )


if __name__ == "__main__":
    main()
