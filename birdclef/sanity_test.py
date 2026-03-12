"""
birdclef.sanity_test — End-to-end shape validation.

Generates synthetic audio data, trains for 2 epochs on 10 samples,
runs inference on a fake soundscape, and validates every tensor shape
along the pipeline. No real data needed.

Usage:
    python -m birdclef.sanity_test
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

SANITY_SPECIES = ["species_a", "species_b", "species_c", "species_d", "species_e"]
NUM_TRAIN = 10
NUM_EPOCHS = 2
SR = 32000


def _make_sine(duration: float, freq: float, sr: int = SR) -> np.ndarray:
    """Generate a sine wave at a given frequency."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def build_synthetic_dataset(data_dir: Path):
    """
    Create a minimal BirdCLEF-style directory structure with:
      - train_audio/<species>/<file>.wav  (10 files, 5-8 sec each)
      - train_metadata.csv
      - taxonomy.csv
      - test_soundscapes/soundscape_001.wav  (30-second file)
      - sample_submission.csv
    """
    train_audio = data_dir / "train_audio"
    test_dir = data_dir / "test_soundscapes"
    train_audio.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    freqs = [440, 880, 660, 550, 770]  # Different freq per species

    for i in range(NUM_TRAIN):
        sp = SANITY_SPECIES[i % len(SANITY_SPECIES)]
        sp_dir = train_audio / sp
        sp_dir.mkdir(exist_ok=True)

        fname = f"{sp}_{i:03d}.wav"
        duration = 5.0 + (i % 4)  # 5-8 seconds
        freq = freqs[i % len(freqs)] + i * 10  # Slight variation

        audio = _make_sine(duration, freq)
        sf.write(sp_dir / fname, audio, SR)

        rows.append({
            "filename": f"{sp}/{fname}",
            "primary_label": sp,
            "secondary_labels": "[]",
            "latitude": 18.0 + i * 0.01,
            "longitude": -66.0 + i * 0.01,
            "rating": 4.0,
        })

    # train_metadata.csv
    meta = pd.DataFrame(rows)
    meta.to_csv(data_dir / "train_metadata.csv", index=False)
    print(f"  Created train_metadata.csv ({len(meta)} rows)")

    # taxonomy.csv
    tax = pd.DataFrame({
        "species_code": SANITY_SPECIES,
        "common_name": [f"Fake {s.title()}" for s in SANITY_SPECIES],
    })
    tax.to_csv(data_dir / "taxonomy.csv", index=False)
    print(f"  Created taxonomy.csv ({len(tax)} species)")

    # test soundscape — 30 seconds of mixed tones
    soundscape = np.concatenate([_make_sine(5.0, f) for f in freqs + [500]])
    sf.write(test_dir / "soundscape_001.wav", soundscape, SR)
    print(f"  Created test soundscape (30s)")

    # sample_submission.csv — 5-sec grid → 6 rows
    sub_rows = []
    for end_sec in range(5, 35, 5):
        row = {"row_id": f"soundscape_001_{end_sec}"}
        for sp in SANITY_SPECIES:
            row[sp] = 0.0
        sub_rows.append(row)
    pd.DataFrame(sub_rows).to_csv(data_dir / "sample_submission.csv", index=False)
    print(f"  Created sample_submission.csv ({len(sub_rows)} rows)")


def run_sanity_test():
    """Full end-to-end sanity check."""
    print("=" * 60)
    print("  BirdCLEF Sanity Test — Synthetic Data")
    print("=" * 60)
    t_start = time.time()

    # ── Step 1: Build synthetic data ───────────────────────────────
    tmp_root = Path(tempfile.mkdtemp(prefix="birdclef_sanity_"))
    data_dir = tmp_root / "birdclef-2026"
    model_dir = tmp_root / "models"
    output_dir = tmp_root / "output"
    model_dir.mkdir()
    output_dir.mkdir()

    print("\n[1/6] Building synthetic dataset...")
    build_synthetic_dataset(data_dir)

    # ── Step 2: Feature extraction shapes ──────────────────────────
    print("\n[2/6] Validating feature extraction shapes...")
    from birdclef.features import (
        load_audio_window, audio_to_melspec, melspec_to_tensor,
        extract_classical_features, iter_soundscape_windows,
        pool_overlapping_predictions,
    )

    test_wav = list((data_dir / "train_audio").rglob("*.wav"))[0]
    y = load_audio_window(test_wav, duration=5.0)
    assert y.shape == (SR * 5,), f"Audio shape: {y.shape} (expected {(SR * 5,)})"
    print(f"  load_audio_window: {y.shape} ✓")

    mel = audio_to_melspec(y)
    assert mel.ndim == 2 and mel.shape[0] == 128, f"Mel shape: {mel.shape}"
    print(f"  audio_to_melspec:  {mel.shape} ✓")

    tensor = melspec_to_tensor(mel)
    assert tensor.shape == (3, 224, 224), f"Tensor shape: {tensor.shape}"
    print(f"  melspec_to_tensor: {tensor.shape} ✓")

    feats = extract_classical_features(y)
    assert feats is not None and len(feats) == 53, f"Classical features: {len(feats) if feats is not None else 'None'}"
    print(f"  classical_features: ({len(feats)},) ✓")

    # ── Step 3: Overlap windowing ──────────────────────────────────
    print("\n[3/6] Validating overlap windowing...")
    soundscape_path = data_dir / "test_soundscapes" / "soundscape_001.wav"
    windows = list(iter_soundscape_windows(soundscape_path))
    print(f"  Soundscape windows (50% overlap): {len(windows)}")

    for i, (chunk, s_time, e_time) in enumerate(windows):
        assert chunk.shape == (SR * 5,), f"Window {i} shape: {chunk.shape}"
        assert e_time - s_time == 5.0, f"Window {i} duration: {e_time - s_time}"
    print(f"  All windows: shape=({SR * 5},), duration=5.0s ✓")

    # Verify overlap pooling collapses back to grid
    fake_preds = np.random.rand(len(windows), len(SANITY_SPECIES)).astype(np.float32)
    start_times = [w[1] for w in windows]
    end_times = [w[2] for w in windows]
    grid_ends, pooled = pool_overlapping_predictions(start_times, end_times, fake_preds)
    print(f"  Pooled: {len(windows)} windows → {len(grid_ends)} grid cells ✓")
    assert pooled.shape == (len(grid_ends), len(SANITY_SPECIES))
    print(f"  Pooled shape: {pooled.shape} ✓")

    # ── Step 4: Model forward pass ─────────────────────────────────
    print("\n[4/6] Validating model forward pass...")
    from birdclef.model import BirdSmallCNN, BirdClassifier
    num_sp = len(SANITY_SPECIES)

    model = BirdSmallCNN(num_species=num_sp)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, num_sp), f"Logits shape: {logits.shape}"
    probs = torch.sigmoid(logits)
    assert probs.min() >= 0 and probs.max() <= 1
    print(f"  SmallCNN forward: input {x.shape} → logits {logits.shape} → probs [0,1] ✓")

    # ── Step 5: Training loop (2 epochs, 10 samples) ──────────────
    print("\n[5/6] Training: 2 epochs × 10 samples (SmallCNN)...")
    from birdclef.train import BirdCLEFDataset, mixup_batch

    meta = pd.read_csv(data_dir / "train_metadata.csv")
    labels = sorted(meta["primary_label"].unique().tolist())
    assert len(labels) == num_sp, f"Label count: {len(labels)}"

    train_ds = BirdCLEFDataset(
        meta, data_dir / "train_audio", labels, augment=False,
    )
    assert len(train_ds) == NUM_TRAIN, f"Dataset size: {len(train_ds)}"

    # Verify single sample
    sample_x, sample_y = train_ds[0]
    assert sample_x.shape == (3, 224, 224), f"Sample input: {sample_x.shape}"
    assert sample_y.shape == (num_sp,), f"Sample target: {sample_y.shape}"
    assert sample_y.sum() >= 1.0, "Target should have at least one positive label"
    print(f"  Dataset: {len(train_ds)} samples, input={sample_x.shape}, target={sample_y.shape} ✓")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=0,
    )

    model = BirdSmallCNN(num_species=num_sp)
    device = "cpu"
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            assert bx.ndim == 4 and bx.shape[1:] == (3, 224, 224)
            assert by.ndim == 2 and by.shape[1] == num_sp

            bx, by = mixup_batch(bx, by, alpha=0.4)

            optimizer.zero_grad()
            logits = model(bx)
            assert logits.shape == by.shape, f"Logits {logits.shape} != target {by.shape}"
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}/{NUM_EPOCHS}: loss={avg_loss:.4f} ({n_batches} batches) ✓")

    # Save model + labels for inference step
    torch.save(model.state_dict(), model_dir / "birdclef_model.pt")
    with open(model_dir / "birdclef_labels.json", "w") as f:
        json.dump(labels, f)
    print(f"  Model saved to {model_dir} ✓")

    # ── Step 6: Inference on test soundscape ───────────────────────
    print("\n[6/6] Running inference on test soundscape...")
    clf = BirdClassifier(backbone="small", model_dir=model_dir)
    clf.load()
    assert clf.is_loaded
    assert len(clf.labels) == num_sp
    print(f"  Model loaded: {num_sp} species ✓")

    # Manual inference to validate shapes at each step
    all_windows = []
    all_start_times = []
    all_end_times = []
    for chunk, s_t, e_t in iter_soundscape_windows(soundscape_path):
        all_windows.append(chunk)
        all_start_times.append(s_t)
        all_end_times.append(e_t)

    batch_tensors = []
    for w in all_windows:
        mel = audio_to_melspec(w)
        t = melspec_to_tensor(mel)
        batch_tensors.append(t)

    batch = torch.stack(batch_tensors)
    assert batch.shape == (len(all_windows), 3, 224, 224)
    print(f"  Inference batch: {batch.shape} ✓")

    raw_probs = clf.predict_batch(batch)
    assert raw_probs.shape == (len(all_windows), num_sp)
    print(f"  Raw predictions: {raw_probs.shape} ✓")

    grid_ends, pooled = pool_overlapping_predictions(
        all_start_times, all_end_times, raw_probs.numpy(),
    )
    print(f"  Pooled to grid: {pooled.shape} ({len(grid_ends)} cells) ✓")

    # Build submission DataFrame
    rows = []
    for g_idx, end_sec in enumerate(grid_ends):
        row = {"row_id": f"soundscape_001_{end_sec}"}
        for sp_idx, sp in enumerate(labels):
            row[sp] = round(float(pooled[g_idx, sp_idx]), 6)
        rows.append(row)

    submission = pd.DataFrame(rows)
    assert "row_id" in submission.columns
    assert len(submission.columns) == 1 + num_sp
    assert all(0.0 <= submission[sp].min() <= submission[sp].max() <= 1.0 for sp in labels)
    print(f"  Submission: {submission.shape} ✓")
    print(f"\n  Preview:\n{submission.to_string(index=False)}")

    # ── Cleanup ────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    shutil.rmtree(tmp_root, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"  ALL SHAPES VALIDATED — end-to-end pipeline OK")
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    run_sanity_test()
