"""
birdclef.perch_embed — Extract bird embeddings using Google Perch.

Google Perch is a bird vocalization model pretrained on millions of
Xeno-Canto recordings. It outputs 1280-dim embeddings per 5s window.

This module handles:
  1. Downloading Perch from TF Hub and saving as SavedModel
  2. Extracting embeddings from training audio
  3. Saving embeddings as numpy arrays for classifier training

Usage on Colab (with internet):
  # Step 1: Download and save Perch model
  python -m birdclef.perch_embed --save-model --model-dir perch_saved_model

  # Step 2: Extract embeddings from training data
  python -m birdclef.perch_embed --extract \
      --model-dir perch_saved_model \
      --train-csv /content/cibuco-boriken/data/birdclef-2026/train.csv \
      --audio-dir /content/cibuco-boriken/data/birdclef-2026/train_audio \
      --output-dir perch_embeddings

Requirements: tensorflow, tensorflow-hub
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("birdclef.perch_embed")

# Perch expects 32kHz mono audio
PERCH_SAMPLE_RATE = 32000
PERCH_WINDOW_SEC = 5.0
PERCH_EMBEDDING_DIM = 1280


def download_and_save_perch(model_dir: str) -> None:
    """Download Perch from TF Hub and save as local SavedModel."""
    import tensorflow_hub as hub

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Perch from TF Hub...")
    model = hub.load("https://tfhub.dev/google/bird-vocalization-classifier/1")

    # Save as SavedModel for offline use on Kaggle
    import tensorflow as tf
    tf.saved_model.save(model, str(model_dir))
    logger.info(f"Perch saved to {model_dir}")


def load_perch_model(model_dir: str):
    """Load a previously saved Perch model."""
    import tensorflow as tf

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Perch model not found at {model_dir}. "
            "Run with --save-model first."
        )

    logger.info(f"Loading Perch from {model_dir}...")
    model = tf.saved_model.load(str(model_dir))
    logger.info("Perch loaded")
    return model


def extract_embedding(model, audio_np: np.ndarray) -> np.ndarray:
    """
    Extract a 1280-dim embedding from a 5-second audio clip.

    Args:
        model: Loaded Perch TF model
        audio_np: float32 numpy array, shape (samples,), 32kHz mono

    Returns:
        np.ndarray of shape (1280,) — the embedding vector
    """
    import tensorflow as tf

    window_samples = int(PERCH_SAMPLE_RATE * PERCH_WINDOW_SEC)

    # Pad or truncate to exactly 5 seconds
    if len(audio_np) < window_samples:
        audio_np = np.pad(audio_np, (0, window_samples - len(audio_np)))
    elif len(audio_np) > window_samples:
        audio_np = audio_np[:window_samples]

    # Perch expects (batch, samples) float32
    waveform = tf.constant(audio_np[np.newaxis, :], dtype=tf.float32)

    # Run model — output format varies by TF Hub version
    # Try the callable signatures Perch exposes
    if hasattr(model, 'infer_tf'):
        output = model.infer_tf(waveform)
    elif hasattr(model, 'front_end'):
        output = model.front_end(waveform)
    else:
        output = model(waveform)

    # Handle dict output (newer versions) or tuple output (older versions)
    if isinstance(output, dict):
        embedding = output["embedding"].numpy().squeeze()
    elif isinstance(output, (tuple, list)):
        # Perch typically returns (logits, embedding) — embedding is the larger one
        arrays = [o.numpy().squeeze() for o in output]
        embedding = max(arrays, key=lambda a: a.size)
    else:
        embedding = output.numpy().squeeze()

    return embedding


def extract_all_embeddings(
    model,
    train_csv: str,
    audio_dir: str,
    output_dir: str,
    max_windows_per_file: int = 5,
) -> None:
    """
    Extract Perch embeddings from all training audio files.

    Saves:
      - embeddings.npy: (N, 1280) float32
      - labels.npy: (N, num_species) float32 multilabel
      - species.json: list of species names
      - manifest.csv: mapping of embedding index to file + species
    """
    import json
    import librosa
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)
    species_list = sorted(df["primary_label"].unique().tolist())
    species_to_idx = {sp: i for i, sp in enumerate(species_list)}
    num_species = len(species_list)

    logger.info(f"Extracting embeddings: {len(df)} files, {num_species} species")

    embeddings = []
    labels = []
    manifest_rows = []
    window_samples = int(PERCH_SAMPLE_RATE * PERCH_WINDOW_SEC)

    t0 = time.time()
    for i, row in df.iterrows():
        if i > 0 and i % 500 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            logger.info(f"  {i}/{len(df)} files ({rate:.1f} files/s)")

        fname = row["filename"]
        audio_path = Path(audio_dir) / fname

        if not audio_path.exists():
            continue

        try:
            audio, _ = librosa.load(str(audio_path), sr=PERCH_SAMPLE_RATE, mono=True)
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            continue

        # Build multilabel target
        label_vec = np.zeros(num_species, dtype=np.float32)
        label_vec[species_to_idx[row["primary_label"]]] = 1.0

        # Handle secondary labels
        if "secondary_labels" in row and pd.notna(row["secondary_labels"]):
            sec = row["secondary_labels"]
            if isinstance(sec, str) and sec.startswith("["):
                try:
                    sec_list = json.loads(sec.replace("'", '"'))
                    for s in sec_list:
                        if s in species_to_idx:
                            label_vec[species_to_idx[s]] = 0.5
                except (json.JSONDecodeError, ValueError):
                    pass

        # Extract windows (up to max_windows_per_file)
        total_samples = len(audio)
        num_windows = min(
            max_windows_per_file,
            max(1, total_samples // window_samples),
        )

        for w in range(num_windows):
            start = w * window_samples
            end = start + window_samples
            chunk = audio[start:end]

            if len(chunk) < window_samples // 2:
                continue

            emb = extract_embedding(model, chunk)
            embeddings.append(emb)
            labels.append(label_vec)
            manifest_rows.append({
                "idx": len(embeddings) - 1,
                "filename": fname,
                "primary_label": row["primary_label"],
                "window": w,
            })

    elapsed = time.time() - t0
    logger.info(f"Extraction complete: {len(embeddings)} embeddings in {elapsed:.1f}s")

    # Save
    embeddings_arr = np.stack(embeddings).astype(np.float32)
    labels_arr = np.stack(labels).astype(np.float32)

    np.save(output_dir / "embeddings.npy", embeddings_arr)
    np.save(output_dir / "labels.npy", labels_arr)

    with open(output_dir / "species.json", "w") as f:
        json.dump(species_list, f)

    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)

    logger.info(f"Saved: {output_dir}")
    logger.info(f"  embeddings.npy: {embeddings_arr.shape}")
    logger.info(f"  labels.npy:     {labels_arr.shape}")
    logger.info(f"  species.json:   {num_species} species")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Perch Embedding Extraction")
    parser.add_argument("--save-model", action="store_true",
                        help="Download Perch from TF Hub and save locally")
    parser.add_argument("--extract", action="store_true",
                        help="Extract embeddings from training audio")
    parser.add_argument("--model-dir", type=str, default="perch_saved_model",
                        help="Directory for saved Perch model")
    parser.add_argument("--train-csv", type=str, default=None,
                        help="Path to train.csv")
    parser.add_argument("--audio-dir", type=str, default=None,
                        help="Path to train_audio/")
    parser.add_argument("--output-dir", type=str, default="perch_embeddings",
                        help="Directory for extracted embeddings")
    parser.add_argument("--max-windows", type=int, default=5,
                        help="Max 5s windows per audio file")
    args = parser.parse_args()

    if args.save_model:
        download_and_save_perch(args.model_dir)

    if args.extract:
        if not args.train_csv or not args.audio_dir:
            print("ERROR: --extract requires --train-csv and --audio-dir")
            sys.exit(1)

        model = load_perch_model(args.model_dir)
        extract_all_embeddings(
            model=model,
            train_csv=args.train_csv,
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            max_windows_per_file=args.max_windows,
        )


if __name__ == "__main__":
    main()
