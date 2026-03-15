"""
birdclef.precompute — Pre-compute mel-spectrogram tensors from audio.

Converts audio windows (from smart_crop manifest or train.csv) into
saved .pt tensor files, eliminating librosa I/O from the training loop.

Usage:
    # From smart crop manifest (recommended):
    python -m birdclef.precompute \
        --manifest birdclef/output/smart_crop_manifest.csv \
        --output-dir birdclef/output/precomputed

    # From train.csv (random 5s window per recording):
    python -m birdclef.precompute \
        --output-dir birdclef/output/precomputed

Output structure:
    output_dir/
        tensors/          — one .pt file per window (3×224×224 float16)
        manifest.csv      — index mapping tensor files to labels
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from birdclef.config import (
    TRAIN_AUDIO_DIR, TRAIN_META_CSV, OUTPUT_DIR,
    SAMPLE_RATE, WINDOW_SECONDS,
)
from birdclef.features import load_audio_window, audio_to_melspec, melspec_to_tensor

logger = logging.getLogger("birdclef.precompute")


def precompute_from_manifest(
    manifest_path: Path,
    train_csv_path: Path,
    audio_dir: Path,
    output_dir: Path,
    max_duration: float = WINDOW_SECONDS,
) -> pd.DataFrame:
    """
    Convert all smart-crop windows into saved tensor files.

    Each window becomes a .pt file (float16 to save disk space).
    Returns a manifest DataFrame mapping tensor filenames to metadata.
    """
    manifest = pd.read_csv(manifest_path)
    train_meta = pd.read_csv(train_csv_path)

    # Build filename → secondary_labels lookup
    sec_lookup: dict[str, str] = {}
    if "secondary_labels" in train_meta.columns:
        for _, r in train_meta[["filename", "secondary_labels"]].drop_duplicates("filename").iterrows():
            sec_lookup[r["filename"]] = str(r["secondary_labels"])

    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(manifest)
    t0 = time.time()
    failed = 0

    for i, (_, row) in enumerate(manifest.iterrows()):
        filename = row["filename"]
        species = row["species"]
        offset = float(row["offset_seconds"])
        confidence = float(row.get("confidence", 0.0))
        window_index = int(row.get("window_index", i))

        # Tensor filename: species__basename__winN.pt
        base = Path(filename).stem
        sp_prefix = Path(filename).parent.name
        tensor_name = f"{sp_prefix}__{base}__win{window_index}.pt"
        tensor_path = tensor_dir / tensor_name

        # Skip if already computed (supports resume)
        if tensor_path.exists():
            rows.append({
                "tensor_file": tensor_name,
                "filename": filename,
                "species": species,
                "offset_seconds": offset,
                "confidence": confidence,
                "secondary_labels": sec_lookup.get(filename, "[]"),
            })
            continue

        try:
            y = load_audio_window(
                audio_dir / filename,
                offset_seconds=offset,
                duration=max_duration,
            )
            mel = audio_to_melspec(y)
            tensor = melspec_to_tensor(mel)
            # Save as float16 to halve disk usage (~300KB → ~150KB per file)
            torch.save(tensor.half(), tensor_path)
        except Exception as e:
            logger.warning(f"[{i}/{total}] Failed {filename}@{offset}s: {e}")
            failed += 1
            continue

        rows.append({
            "tensor_file": tensor_name,
            "filename": filename,
            "species": species,
            "offset_seconds": offset,
            "confidence": confidence,
            "secondary_labels": sec_lookup.get(filename, "[]"),
        })

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(
                f"  [{i+1:>6d}/{total}] "
                f"{rate:.1f} windows/sec | "
                f"ETA: {eta/60:.1f} min"
            )

    out_df = pd.DataFrame(rows)
    manifest_out = output_dir / "manifest.csv"
    out_df.to_csv(manifest_out, index=False)

    elapsed = time.time() - t0
    print(f"\nPrecompute complete:")
    print(f"  Tensors: {len(out_df)} saved to {tensor_dir}")
    print(f"  Failed:  {failed}")
    print(f"  Time:    {elapsed/60:.1f} min")
    print(f"  Manifest: {manifest_out}")

    return out_df


def precompute_from_train_csv(
    train_csv_path: Path,
    audio_dir: Path,
    output_dir: Path,
    max_duration: float = WINDOW_SECONDS,
) -> pd.DataFrame:
    """
    Convert train.csv recordings into saved tensor files.

    Takes one random 5s window per recording (offset=0).
    """
    meta = pd.read_csv(train_csv_path)
    label_col = "primary_label" if "primary_label" in meta.columns else "species"

    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(meta)
    t0 = time.time()
    failed = 0

    for i, (_, row) in enumerate(meta.iterrows()):
        filename = row["filename"]
        species = row[label_col]
        secondary = str(row.get("secondary_labels", "[]"))

        base = Path(filename).stem
        sp_prefix = Path(filename).parent.name
        tensor_name = f"{sp_prefix}__{base}__win0.pt"
        tensor_path = tensor_dir / tensor_name

        if tensor_path.exists():
            rows.append({
                "tensor_file": tensor_name,
                "filename": filename,
                "species": species,
                "offset_seconds": 0.0,
                "confidence": 1.0,
                "secondary_labels": secondary,
            })
            continue

        try:
            y = load_audio_window(
                audio_dir / filename,
                offset_seconds=0.0,
                duration=max_duration,
            )
            mel = audio_to_melspec(y)
            tensor = melspec_to_tensor(mel)
            torch.save(tensor.half(), tensor_path)
        except Exception as e:
            logger.warning(f"[{i}/{total}] Failed {filename}: {e}")
            failed += 1
            continue

        rows.append({
            "tensor_file": tensor_name,
            "filename": filename,
            "species": species,
            "offset_seconds": 0.0,
            "confidence": 1.0,
            "secondary_labels": secondary,
        })

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(
                f"  [{i+1:>6d}/{total}] "
                f"{rate:.1f} windows/sec | "
                f"ETA: {eta/60:.1f} min"
            )

    out_df = pd.DataFrame(rows)
    manifest_out = output_dir / "manifest.csv"
    out_df.to_csv(manifest_out, index=False)

    elapsed = time.time() - t0
    print(f"\nPrecompute complete:")
    print(f"  Tensors: {len(out_df)} saved to {tensor_dir}")
    print(f"  Failed:  {failed}")
    print(f"  Time:    {elapsed/60:.1f} min")
    print(f"  Manifest: {manifest_out}")

    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute mel-spectrogram tensors for fast training"
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to smart_crop_manifest.csv (if omitted, uses train.csv directly)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(OUTPUT_DIR / "precomputed"),
        help="Directory to save tensor files and manifest",
    )
    parser.add_argument(
        "--audio-dir", type=str, default=None,
        help="Override audio directory (default: auto-detect from config)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    audio_dir = Path(args.audio_dir) if args.audio_dir else TRAIN_AUDIO_DIR

    # Find train.csv
    train_csv = TRAIN_META_CSV
    if not train_csv.exists():
        alt = train_csv.parent / "train.csv"
        if alt.exists():
            train_csv = alt

    print("=" * 60)
    print("  BirdCLEF 2026 — Spectrogram Pre-computation")
    print(f"  Audio dir:  {audio_dir}")
    print(f"  Output dir: {output_dir}")
    if args.manifest:
        print(f"  Manifest:   {args.manifest}")
    else:
        print(f"  Source:     {train_csv}")
    print("=" * 60)

    if args.manifest:
        precompute_from_manifest(
            manifest_path=Path(args.manifest),
            train_csv_path=train_csv,
            audio_dir=audio_dir,
            output_dir=output_dir,
        )
    else:
        precompute_from_train_csv(
            train_csv_path=train_csv,
            audio_dir=audio_dir,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
