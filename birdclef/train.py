"""
birdclef.train — Training pipeline for BirdCLEF 2026.

Trains a multilabel CNN on BirdCLEF training audio.
Supports mixup augmentation and multi-backbone selection.

Usage (local):
    python -m birdclef.train --backbone efficientnet_b0 --epochs 30
    python -m birdclef.train --backbone small --epochs 10 --fast
    python -m birdclef.train --backbone small --loss focal --epochs 10 --fast
    python -m birdclef.train --smart-crop output/smart_crop_manifest.csv --epochs 30
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from birdclef.config import (
    TRAIN_AUDIO_DIR, TRAIN_META_CSV, MODEL_DIR, OUTPUT_DIR,
    TRAIN_SOUNDSCAPES_DIR, TRAIN_SOUNDSCAPES_LABELS_CSV,
    MODEL_FILENAME, LABELS_FILENAME, TRAINING_METADATA_FILENAME,
    SAMPLE_RATE, WINDOW_SECONDS,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, TRAIN_SPLIT, MIXUP_ALPHA,
)
from birdclef.features import load_audio_window, audio_to_melspec, melspec_to_tensor, spec_augment
from birdclef.model import BACKBONE_BUILDERS

logger = logging.getLogger("birdclef.train")


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

class BirdCLEFDataset(Dataset):
    """
    Loads training audio clips, extracts mel-spectrograms on-the-fly.

    Each sample: (mel_tensor, multilabel_target)
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        audio_dir: Path,
        labels: List[str],
        max_duration: float = WINDOW_SECONDS,
        augment: bool = False,
        secondary_weight: float = 0.5,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        self.num_labels = len(labels)
        self.max_duration = max_duration
        self.augment = augment
        self.secondary_weight = secondary_weight

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        filepath = self.audio_dir / row["filename"]

        # Random offset for augmentation during training
        offset = 0.0
        if self.augment:
            try:
                import librosa
                file_duration = librosa.get_duration(path=filepath)
                max_offset = max(0, file_duration - self.max_duration)
                offset = random.uniform(0, max_offset) if max_offset > 0 else 0.0
            except Exception:
                offset = 0.0

        try:
            y = load_audio_window(filepath, offset_seconds=offset, duration=self.max_duration)
            mel = audio_to_melspec(y)
            if self.augment:
                mel = spec_augment(mel)
            tensor = melspec_to_tensor(mel)
        except Exception as e:
            logger.warning(f"Failed to process {filepath}: {e}. Returning silence.")
            tensor = torch.zeros(3, 224, 224)

        # Build multilabel target
        target = torch.zeros(self.num_labels, dtype=torch.float32)
        species = row.get("primary_label", row.get("species", ""))
        if species in self.label_to_idx:
            target[self.label_to_idx[species]] = 1.0

        # Secondary labels (if provided) — weighted below primary
        secondary = row.get("secondary_labels", "")
        if isinstance(secondary, str) and secondary.strip():
            try:
                sec_labels = json.loads(secondary.replace("'", '"'))
                for s in sec_labels:
                    if s in self.label_to_idx:
                        target[self.label_to_idx[s]] = self.secondary_weight
            except (json.JSONDecodeError, TypeError):
                pass

        return tensor, target


# ═══════════════════════════════════════════════════════════════════
# Smart Crop Dataset (CFAR-filtered windows)
# ═══════════════════════════════════════════════════════════════════

class SmartCropDataset(Dataset):
    """
    Loads pre-selected CFAR-filtered windows from a smart_crop_manifest.csv.

    Each row in the manifest specifies a file + offset, so we load exactly
    the windows that passed the energy CFAR gate — no random offsets.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        audio_dir: Path,
        labels: List[str],
        max_duration: float = WINDOW_SECONDS,
        augment: bool = False,
        secondary_weight: float = 0.5,
        train_meta: pd.DataFrame | None = None,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        self.num_labels = len(labels)
        self.max_duration = max_duration
        self.augment = augment
        self.secondary_weight = secondary_weight
        # Build filename → secondary_labels lookup from original metadata
        self._sec_lookup: dict[str, str] = {}
        if train_meta is not None and "secondary_labels" in train_meta.columns:
            for _, r in train_meta[["filename", "secondary_labels"]].drop_duplicates("filename").iterrows():
                self._sec_lookup[r["filename"]] = str(r["secondary_labels"])

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]
        filepath = self.audio_dir / row["filename"]
        offset = float(row["offset_seconds"])

        try:
            y = load_audio_window(filepath, offset_seconds=offset, duration=self.max_duration)
            mel = audio_to_melspec(y)
            if self.augment:
                mel = spec_augment(mel)
            tensor = melspec_to_tensor(mel)
        except Exception as e:
            logger.warning(f"SmartCrop: failed to process {filepath}@{offset}s: {e}. Returning silence.")
            tensor = torch.zeros(3, 224, 224)

        # Build multilabel target from species column
        target = torch.zeros(self.num_labels, dtype=torch.float32)
        species = row.get("species", "")
        if species in self.label_to_idx:
            target[self.label_to_idx[species]] = 1.0

        # Secondary labels from original metadata
        secondary = self._sec_lookup.get(row.get("filename", ""), "")
        if isinstance(secondary, str) and secondary.strip():
            try:
                sec_labels = json.loads(secondary.replace("'", '"'))
                for s in sec_labels:
                    if s in self.label_to_idx:
                        target[self.label_to_idx[s]] = self.secondary_weight
            except (json.JSONDecodeError, TypeError):
                pass

        return tensor, target


# ═══════════════════════════════════════════════════════════════════
# Soundscape Dataset (P2-00)
# ═══════════════════════════════════════════════════════════════════

class SoundscapeDataset(Dataset):
    """
    Loads labeled 5-sec windows from train_soundscapes.

    train_soundscapes_labels.csv format:
        filename, start (HH:MM:SS), end (HH:MM:SS), primary_label (semicolon-separated)

    Each row is one 5-second window with multilabel annotations.
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        soundscapes_dir: Path,
        label_list: List[str],
    ):
        self.df = labels_df.reset_index(drop=True)
        self.soundscapes_dir = soundscapes_dir
        self.label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}
        self.num_labels = len(label_list)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _hhmmss_to_seconds(t: str) -> float:
        """Convert HH:MM:SS string to seconds."""
        parts = t.strip().split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        filepath = self.soundscapes_dir / row["filename"]
        offset = self._hhmmss_to_seconds(row["start"])

        try:
            y = load_audio_window(filepath, offset_seconds=offset, duration=WINDOW_SECONDS)
            mel = audio_to_melspec(y)
            tensor = melspec_to_tensor(mel)
        except Exception as e:
            logger.warning(f"Failed to process {filepath}@{offset}s: {e}. Returning silence.")
            tensor = torch.zeros(3, 224, 224)

        # Build multilabel target from semicolon-separated species codes
        target = torch.zeros(self.num_labels, dtype=torch.float32)
        raw_label = str(row.get("primary_label", ""))
        for sp in raw_label.split(";"):
            sp = sp.strip()
            if sp in self.label_to_idx:
                target[self.label_to_idx[sp]] = 1.0

        return tensor, target


def load_soundscape_meta(labels: List[str]) -> Tuple[pd.DataFrame, int]:
    """
    Load train_soundscapes_labels.csv if it exists.

    Returns:
        (dataframe, count) — empty DataFrame and 0 if file missing.
    """
    csv_candidates = [
        TRAIN_SOUNDSCAPES_LABELS_CSV,
        TRAIN_SOUNDSCAPES_LABELS_CSV.parent / "train_soundscapes_labels.csv",
    ]
    resolved = next((p for p in csv_candidates if p.exists()), None)
    if resolved is None:
        return pd.DataFrame(), 0

    df = pd.read_csv(resolved)
    print(f"Soundscapes file: {resolved.name}")
    print(f"  Soundscape labeled windows: {len(df)} rows, {df['filename'].nunique()} files")
    return df, len(df)


# ═══════════════════════════════════════════════════════════════════
# Class Weights for Weighted BCE (P2-05)
# ═══════════════════════════════════════════════════════════════════

def compute_class_weights(
    meta: pd.DataFrame, labels: List[str], label_col: str = "primary_label",
) -> torch.Tensor:
    """
    Compute pos_weight = 1 / sqrt(species_count) for BCEWithLogitsLoss.

    Rare species (small count) get higher weight so the model doesn't
    ignore them. Common species (large count) get lower weight.

    Returns:
        1-D tensor of shape (num_species,) with per-class pos_weights.
    """
    counts = meta[label_col].value_counts()
    weights = []
    for sp in labels:
        c = counts.get(sp, 1)  # default 1 to avoid division by zero
        weights.append(1.0 / np.sqrt(max(c, 1)))
    w = torch.tensor(weights, dtype=torch.float32)
    # Normalize so mean weight = 1.0 (keeps loss scale stable)
    w = w / w.mean()
    return w


class MultilabelFocalLoss(nn.Module):
    """
    Standard multilabel focal loss built on BCE-with-logits.

    Args:
        alpha: positive-class balancing factor
        gamma: focal focusing parameter
        reduction: reduction mode for batch aggregation
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)
        loss = focal_weight * bce

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_training_metadata(
    backbone: str,
    loss_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    use_mixup: bool,
    include_soundscapes: bool,
    use_weighted_bce: bool,
    smart_crop: bool = False,
    secondary_weight: float = 0.5,
    best_val_loss: float = float("inf"),
) -> dict:
    """Build a small JSON-serializable summary for the saved checkpoint."""
    metadata = {
        "backbone": backbone,
        "loss": loss_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "mixup": use_mixup,
        "include_soundscapes": include_soundscapes,
        "weighted_bce": use_weighted_bce if loss_name == "bce" else False,
        "smart_crop": smart_crop,
        "secondary_weight": secondary_weight,
        "best_val_loss": best_val_loss,
    }
    if loss_name == "focal":
        metadata["focal_alpha"] = 0.25
        metadata["focal_gamma"] = 2.0
    return metadata


# ═══════════════════════════════════════════════════════════════════
# Mixup Augmentation
# ═══════════════════════════════════════════════════════════════════

def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float = MIXUP_ALPHA,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mixup augmentation to a batch.
    Blends pairs of samples with random lambda from Beta distribution.
    """
    if alpha <= 0:
        return x, y

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y


# ═══════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════

def train(
    backbone: str = "efficientnet_b0",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    loss_name: str = "bce",
    use_mixup: bool = True,
    fast: bool = False,
    max_samples: int | None = None,
    include_soundscapes: bool = False,
    use_weighted_bce: bool = True,
    smart_crop: str | None = None,
    secondary_weight: float = 0.5,
):
    """
    Full training pipeline.

    Args:
        backbone: Model architecture to use
        epochs: Number of training epochs
        batch_size: Batch size for DataLoader
        lr: Learning rate
        loss_name: Loss function to use ("bce" or "focal")
        use_mixup: Whether to apply mixup augmentation
        fast: If True, use only 10% of data (for quick iteration)
    """
    print("=" * 60)
    print("  BirdCLEF 2026 — Training Pipeline")
    print(f"  Backbone: {backbone} | Epochs: {epochs} | Loss: {loss_name}")
    print(f"  Secondary label weight: {secondary_weight}")
    if smart_crop:
        print(f"  Smart Crop: {smart_crop}")
    print("=" * 60)

    # ── Load metadata ──────────────────────────────────────────────
    meta_csv_candidates = [
        TRAIN_META_CSV,
        TRAIN_META_CSV.parent / "train.csv",
    ]
    resolved_meta_csv = next((path for path in meta_csv_candidates if path.exists()), None)
    if resolved_meta_csv is None:
        print(f"ERROR: Training metadata not found. Tried: {meta_csv_candidates}")
        print("Download the competition data first:")
        print("  kaggle competitions download -c birdclef-2026")
        return

    meta = pd.read_csv(resolved_meta_csv)
    print(f"Metadata file: {resolved_meta_csv.name}")
    print(f"Loaded {len(meta)} training samples")

    # ── Build label list (sorted for determinism) ──────────────────
    label_col = "primary_label" if "primary_label" in meta.columns else "species"
    labels = sorted(meta[label_col].unique().tolist())
    num_species = len(labels)
    print(f"Species count: {num_species}")

    # ── Fast mode: subsample ───────────────────────────────────────
    if fast:
        meta = meta.sample(frac=0.1, random_state=42).reset_index(drop=True)
        print(f"Fast mode: using {len(meta)} samples")

    # ── Explicit sample cap for smoke tests ───────────────────────
    if max_samples is not None and max_samples > 0 and len(meta) > max_samples:
        meta = meta.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Max samples mode: using {len(meta)} samples")

    # ── Train/Val split ────────────────────────────────────────────
    split_idx = int(len(meta) * TRAIN_SPLIT)
    train_meta = meta.iloc[:split_idx]
    val_meta = meta.iloc[split_idx:]
    print(f"Train: {len(train_meta)} | Val: {len(val_meta)}")

    # ── Dataset selection: smart-crop vs standard ──────────────────
    if smart_crop:
        manifest_path = Path(smart_crop)
        if not manifest_path.is_absolute():
            manifest_path = OUTPUT_DIR / manifest_path
        if not manifest_path.exists():
            print(f"ERROR: Smart crop manifest not found: {manifest_path}")
            print("Run smart crop first:  python -m birdclef.smart_crop")
            return

        sc_manifest = pd.read_csv(manifest_path)
        print(f"Smart crop manifest: {len(sc_manifest)} pre-filtered windows")

        # Split manifest by filename for train/val
        sc_files = sc_manifest["filename"].unique()
        sc_split_idx = int(len(sc_files) * TRAIN_SPLIT)
        train_files = set(sc_files[:sc_split_idx])
        val_files = set(sc_files[sc_split_idx:])

        train_manifest = sc_manifest[sc_manifest["filename"].isin(train_files)]
        val_manifest = sc_manifest[sc_manifest["filename"].isin(val_files)]

        train_ds = SmartCropDataset(
            train_manifest, TRAIN_AUDIO_DIR, labels,
            augment=True, secondary_weight=secondary_weight, train_meta=meta,
        )
        val_ds = SmartCropDataset(
            val_manifest, TRAIN_AUDIO_DIR, labels,
            augment=False, secondary_weight=secondary_weight, train_meta=meta,
        )
        print(f"Smart crop split: Train {len(train_ds)} | Val {len(val_ds)} windows")
    else:
        train_ds = BirdCLEFDataset(
            train_meta, TRAIN_AUDIO_DIR, labels,
            augment=True, secondary_weight=secondary_weight,
        )
        val_ds = BirdCLEFDataset(
            val_meta, TRAIN_AUDIO_DIR, labels,
            augment=False, secondary_weight=secondary_weight,
        )

    # ── Optionally merge soundscape windows into training set ──────
    if include_soundscapes:
        sc_df, sc_count = load_soundscape_meta(labels)
        if sc_count > 0:
            sc_ds = SoundscapeDataset(sc_df, TRAIN_SOUNDSCAPES_DIR, labels)
            train_ds = torch.utils.data.ConcatDataset([train_ds, sc_ds])
            print(f"  + Added {sc_count} soundscape windows → {len(train_ds)} total train samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Build model ────────────────────────────────────────────────
    builder = BACKBONE_BUILDERS.get(backbone)
    if builder is None:
        print(f"Unknown backbone: {backbone}. Options: {list(BACKBONE_BUILDERS.keys())}")
        return

    model = builder(num_species)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")

    # ── Loss + Optimizer ───────────────────────────────────────────
    if loss_name == "focal":
        criterion = MultilabelFocalLoss(alpha=0.25, gamma=2.0)
        print("Focal loss: alpha=0.25, gamma=2.0")
    elif use_weighted_bce:
        pos_weight = compute_class_weights(meta, labels, label_col).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Weighted BCE: min_w={pos_weight.min():.3f}, max_w={pos_weight.max():.3f}, mean={pos_weight.mean():.3f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Loss: BCEWithLogitsLoss")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    # ── Training ───────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            if use_mixup:
                batch_x, batch_y = mixup_batch(batch_x, batch_y)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train = train_loss / max(len(train_loader), 1)

        # ── Validation ─────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

        avg_val = val_loss / max(len(val_loader), 1)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), MODEL_DIR / MODEL_FILENAME)
            with open(MODEL_DIR / LABELS_FILENAME, "w", encoding="utf-8") as f:
                json.dump(labels, f)
            metadata = build_training_metadata(
                backbone=backbone,
                loss_name=loss_name,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                use_mixup=use_mixup,
                include_soundscapes=include_soundscapes,
                use_weighted_bce=use_weighted_bce,
                smart_crop=smart_crop is not None,
                secondary_weight=secondary_weight,
                best_val_loss=avg_val,
            )
            with open(MODEL_DIR / TRAINING_METADATA_FILENAME, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            print(f"  -> Saved best model (val_loss={avg_val:.4f})")
            print(f"  -> Saved training metadata ({TRAINING_METADATA_FILENAME})")

    print("=" * 60)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_DIR / MODEL_FILENAME}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Training")
    parser.add_argument("--backbone", default="efficientnet_b0",
                        choices=list(BACKBONE_BUILDERS.keys()))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--loss", default="bce", choices=["bce", "focal"],
                        help="Training loss: standard BCE or multilabel focal loss")
    parser.add_argument("--no-mixup", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Use 10%% of data for quick iteration")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional hard cap on total samples (for tiny smoke tests)")
    parser.add_argument("--include-soundscapes", action="store_true",
                        help="Merge train_soundscapes labeled windows into training data")
    parser.add_argument("--no-weighted-bce", action="store_true",
                        help="Disable class-weighted BCE loss")
    parser.add_argument("--smart-crop", type=str, default=None,
                        help="Path to smart_crop_manifest.csv (CFAR-filtered windows)")
    parser.add_argument("--secondary-weight", type=float, default=0.5,
                        help="Weight for secondary species labels (0 to disable, default 0.5)")

    args = parser.parse_args()
    train(
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_name=args.loss,
        use_mixup=not args.no_mixup,
        fast=args.fast,
        max_samples=args.max_samples,
        include_soundscapes=args.include_soundscapes,
        use_weighted_bce=not args.no_weighted_bce,
        smart_crop=args.smart_crop,
        secondary_weight=args.secondary_weight,
    )


if __name__ == "__main__":
    main()
