"""
birdclef.inference — Kaggle-compatible inference pipeline.

Produces submission.csv from test soundscapes.

Designed for:
  - CPU runtime ≤ 90 minutes
  - No internet access
  - Reads model from attached Kaggle dataset

Usage (local test):
    python -m birdclef.inference

On Kaggle:
    Attach trained model as a dataset, then run this in a code cell.
"""

import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from birdclef.config import (
    TEST_AUDIO_DIR, SAMPLE_SUBMISSION, MODEL_DIR, OUTPUT_DIR,
    SAMPLE_RATE, WINDOW_SECONDS, IS_KAGGLE,
)
from birdclef.features import (
    iter_soundscape_windows, audio_to_melspec, melspec_to_tensor,
    pool_overlapping_predictions,
)
from birdclef.model import BirdClassifier

logger = logging.getLogger("birdclef.inference")


def run_inference(
    backbone: str = "efficientnet_b0",
    model_dir: Path = MODEL_DIR,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Run inference on all test soundscapes and produce a submission DataFrame.

    Returns:
        DataFrame matching Kaggle's submission format:
        row_id, species_1, species_2, ..., species_N
    """
    t_start = time.time()

    # ── Load model ─────────────────────────────────────────────────
    clf = BirdClassifier(backbone=backbone, model_dir=model_dir)
    clf.load()
    species_list = clf.labels
    num_species = len(species_list)
    print(f"Model loaded: {backbone}, {num_species} species, device={clf.device}")

    # ── Load sample submission for row_id format ───────────────────
    if SAMPLE_SUBMISSION.exists():
        sample_sub = pd.read_csv(SAMPLE_SUBMISSION)
        # Extract unique soundscape filenames and expected row_ids
        print(f"Sample submission: {len(sample_sub)} rows")
    else:
        sample_sub = None
        print("No sample_submission.csv found — generating row_ids from test audio files")

    # ── Discover test soundscapes ──────────────────────────────────
    test_dir = TEST_AUDIO_DIR
    if not test_dir.exists():
        print(f"WARNING: Test directory not found: {test_dir}")
        print("Creating empty submission with zero probabilities.")
        if sample_sub is not None:
            return sample_sub
        return pd.DataFrame(columns=["row_id"] + species_list)

    soundscape_files = sorted(test_dir.glob("*.ogg"))
    if not soundscape_files:
        soundscape_files = sorted(test_dir.glob("*.wav"))
    if not soundscape_files:
        soundscape_files = sorted(test_dir.glob("*.flac"))

    print(f"Found {len(soundscape_files)} test soundscapes")

    # ── Process each soundscape ────────────────────────────────────
    rows = []

    for file_idx, filepath in enumerate(soundscape_files):
        file_stem = filepath.stem
        print(f"  [{file_idx + 1}/{len(soundscape_files)}] {filepath.name}...", end=" ", flush=True)
        file_t0 = time.time()

        # Collect all overlapping 5-sec windows
        windows = []
        start_times = []
        end_times = []
        for chunk, start_time, end_time in iter_soundscape_windows(filepath):
            windows.append(chunk)
            start_times.append(start_time)
            end_times.append(end_time)

        # Process in batches for memory efficiency
        all_probs = []
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i + batch_size]
            batch_tensors = []
            for w in batch_windows:
                mel = audio_to_melspec(w)
                tensor = melspec_to_tensor(mel)
                batch_tensors.append(tensor)

            batch = torch.stack(batch_tensors)
            probs = clf.predict_batch(batch)  # (B, num_species)
            all_probs.append(probs)

        all_probs = torch.cat(all_probs, dim=0).numpy()  # (num_windows, num_species)

        # Pool overlapping windows back to 5-second submission grid
        grid_end_times, pooled_probs = pool_overlapping_predictions(
            start_times, end_times, all_probs,
        )

        # Build submission rows from pooled predictions
        for g_idx, end_sec in enumerate(grid_end_times):
            row_id = f"{file_stem}_{end_sec}"
            prob_values = pooled_probs[g_idx]

            row = {"row_id": row_id}
            for sp_idx, sp_name in enumerate(species_list):
                row[sp_name] = round(float(prob_values[sp_idx]), 6)
            rows.append(row)

        elapsed = time.time() - file_t0
        print(f"{len(grid_end_times)} grid cells ({len(windows)} windows), {elapsed:.1f}s")

    submission = pd.DataFrame(rows)

    # ── Validate against sample submission ─────────────────────────
    if sample_sub is not None:
        expected_cols = set(sample_sub.columns)
        actual_cols = set(submission.columns)
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols

        if missing_cols:
            # Zero-shot species: use 1/N prior instead of 0.0
            zero_shot_prior = 1.0 / len(expected_cols - {"row_id"})
            print(f"WARNING: {len(missing_cols)} missing species (zero-shot), using prior={zero_shot_prior:.6f}")
            for col in missing_cols:
                submission[col] = zero_shot_prior
        if extra_cols:
            print(f"NOTE: Extra columns (will be ignored): {extra_cols}")

        # Reorder columns to match sample submission
        zero_shot_fill = 1.0 / max(len(expected_cols) - 1, 1)
        submission = submission.reindex(columns=sample_sub.columns, fill_value=zero_shot_fill)

    total_time = time.time() - t_start
    print(f"\nInference complete: {len(submission)} rows in {total_time:.1f}s")
    return submission


def main():
    """CLI entry point for inference."""
    submission = run_inference()

    output_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Submission saved: {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Preview:\n{submission.head()}")


if __name__ == "__main__":
    main()
