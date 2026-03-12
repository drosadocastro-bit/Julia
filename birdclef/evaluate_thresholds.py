"""
birdclef.evaluate_thresholds — Compare fixed vs CFAR adaptive thresholds.

Runs the trained model on the validation split, collects sigmoid
probabilities, then evaluates both thresholding strategies:

  1. Fixed threshold (t=0.5)
  2. CFAR adaptive threshold (k=2.0)

Reports:
  - Macro ROC-AUC (threshold-free, competition metric)
  - Rare-species F1 (species with <50 training clips)
  - False positive rate on soundscape windows

Usage:
    python -m birdclef.evaluate_thresholds --backbone small
    python -m birdclef.evaluate_thresholds --backbone small --include-soundscapes
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from birdclef.config import (
    TRAIN_AUDIO_DIR, TRAIN_META_CSV, MODEL_DIR,
    TRAIN_SOUNDSCAPES_DIR, TRAIN_SOUNDSCAPES_LABELS_CSV,
    MODEL_FILENAME, LABELS_FILENAME,
    SAMPLE_RATE, WINDOW_SECONDS, BATCH_SIZE, TRAIN_SPLIT,
)
from birdclef.model import BACKBONE_BUILDERS
from birdclef.train import BirdCLEFDataset, SoundscapeDataset, load_soundscape_meta
from birdclef.cfar_threshold import cfar_adaptive_threshold, fixed_threshold, apply_threshold

logger = logging.getLogger("birdclef.evaluate_thresholds")


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model on a DataLoader, return (probs, targets) as numpy arrays.

    Returns:
        probs:   (N, num_species) sigmoid probabilities
        targets: (N, num_species) ground truth binary labels
    """
    all_probs = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(batch_y.numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)


def macro_roc_auc(targets: np.ndarray, probs: np.ndarray) -> float:
    """
    Macro-averaged ROC-AUC (competition metric).
    Skips species with no positive or no negative examples.
    """
    from sklearn.metrics import roc_auc_score

    num_species = targets.shape[1]
    aucs = []
    for i in range(num_species):
        if targets[:, i].sum() == 0 or targets[:, i].sum() == len(targets[:, i]):
            continue  # skip species with no variation
        try:
            auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(auc)
        except ValueError:
            continue

    return float(np.mean(aucs)) if aucs else 0.0


def species_f1(
    targets: np.ndarray,
    preds: np.ndarray,
    species_indices: List[int],
) -> float:
    """
    Macro F1 over a subset of species (e.g., rare ones).
    """
    from sklearn.metrics import f1_score

    if not species_indices:
        return 0.0

    t_sub = targets[:, species_indices]
    p_sub = preds[:, species_indices]

    # Per-species F1, then average
    f1s = []
    for i in range(t_sub.shape[1]):
        if t_sub[:, i].sum() == 0 and p_sub[:, i].sum() == 0:
            f1s.append(1.0)  # both empty = correct
        elif t_sub[:, i].sum() == 0 or p_sub[:, i].sum() == 0:
            f1s.append(0.0)
        else:
            f1s.append(float(f1_score(t_sub[:, i], p_sub[:, i], zero_division=0)))

    return float(np.mean(f1s)) if f1s else 0.0


def false_positive_rate(targets: np.ndarray, preds: np.ndarray) -> float:
    """
    Overall false positive rate: FP / (FP + TN).
    """
    negatives = (targets == 0)
    if negatives.sum() == 0:
        return 0.0
    fp = ((preds == 1) & negatives).sum()
    return float(fp / negatives.sum())


def evaluate(
    backbone: str = "small",
    k: float = 2.0,
    fixed_t: float = 0.5,
    batch_size: int = BATCH_SIZE,
    max_samples: int | None = None,
    include_soundscapes: bool = False,
    verbose: bool = True,
):
    """
    Main evaluation: load model, run val set, compare thresholds.
    """
    if verbose:
        print("=" * 60)
        print("  BirdCLEF 2026 — Threshold Evaluation")
        print(f"  Backbone: {backbone} | Fixed t={fixed_t} | CFAR k={k}")
        print("=" * 60)

    # ── Load metadata ──────────────────────────────────────────────
    meta_csv_candidates = [TRAIN_META_CSV, TRAIN_META_CSV.parent / "train.csv"]
    resolved = next((p for p in meta_csv_candidates if p.exists()), None)
    if resolved is None:
        print("ERROR: Training metadata not found.")
        return None

    meta = pd.read_csv(resolved)
    label_col = "primary_label" if "primary_label" in meta.columns else "species"
    labels = sorted(meta[label_col].unique().tolist())
    num_species = len(labels)
    if verbose:
        print(f"Species: {num_species}")

    # ── Sample cap for quick eval ──────────────────────────────────
    if max_samples is not None and max_samples > 0 and len(meta) > max_samples:
        meta = meta.sample(n=max_samples, random_state=42).reset_index(drop=True)
        if verbose:
            print(f"Max samples: using {len(meta)}")

    # ── Val split (same as training) ───────────────────────────────
    split_idx = int(len(meta) * TRAIN_SPLIT)
    val_meta = meta.iloc[split_idx:]
    if verbose:
        print(f"Validation samples: {len(val_meta)}")

    val_ds = BirdCLEFDataset(val_meta, TRAIN_AUDIO_DIR, labels, augment=False)

    # ── Optionally add soundscape windows as extra eval data ───────
    sc_ds = None
    if include_soundscapes:
        sc_df, sc_count = load_soundscape_meta(labels)
        if sc_count > 0:
            sc_ds = SoundscapeDataset(sc_df, TRAIN_SOUNDSCAPES_DIR, labels)
            if verbose:
                print(f"Soundscape eval windows: {sc_count}")

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Load trained model ─────────────────────────────────────────
    model_path = MODEL_DIR / MODEL_FILENAME
    labels_path = MODEL_DIR / LABELS_FILENAME

    if not model_path.exists():
        print(f"ERROR: No trained model at {model_path}")
        print("Run training first: python -m birdclef.train")
        return None

    builder = BACKBONE_BUILDERS.get(backbone)
    if builder is None:
        print(f"Unknown backbone: {backbone}")
        return None

    model = builder(num_species)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    if verbose:
        print(f"Device: {device}")
        print(f"Model loaded: {model_path.name}")

    # ── Collect validation predictions ─────────────────────────────
    if verbose:
        print("\nCollecting validation predictions...")
    val_probs, val_targets = collect_predictions(model, val_loader, device)
    if verbose:
        print(f"  Val matrix: {val_probs.shape}")

    # ── Also collect soundscape predictions if requested ───────────
    sc_probs, sc_targets = None, None
    if sc_ds is not None:
        sc_loader = DataLoader(sc_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        if verbose:
            print("Collecting soundscape predictions...")
        sc_probs, sc_targets = collect_predictions(model, sc_loader, device)
        if verbose:
            print(f"  Soundscape matrix: {sc_probs.shape}")

    # ── Identify rare species (< 50 training clips) ───────────────
    train_meta = meta.iloc[:split_idx]
    species_counts = train_meta[label_col].value_counts()
    rare_species = [s for s in labels if species_counts.get(s, 0) < 50]
    rare_indices = [labels.index(s) for s in rare_species]
    if verbose:
        print(f"\nRare species (<50 clips): {len(rare_species)}/{num_species}")

    # ── Compute thresholds ─────────────────────────────────────────
    t_fixed = fixed_threshold(val_probs, t=fixed_t)
    t_cfar = cfar_adaptive_threshold(val_probs, k=k)

    preds_fixed = apply_threshold(val_probs, t_fixed)
    preds_cfar = apply_threshold(val_probs, t_cfar)

    # ── Metrics ────────────────────────────────────────────────────
    auc_val = macro_roc_auc(val_targets, val_probs)

    f1_fixed_rare = species_f1(val_targets, preds_fixed, rare_indices)
    f1_cfar_rare = species_f1(val_targets, preds_cfar, rare_indices)

    fpr_fixed_val = false_positive_rate(val_targets, preds_fixed)
    fpr_cfar_val = false_positive_rate(val_targets, preds_cfar)

    # ── Print report ───────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("  RESULTS")
        print("=" * 60)

        print(f"\n{'Metric':<35} {f'Fixed (t={fixed_t})':<18} {f'CFAR (k={k})':<18}")
        print("-" * 71)
        print(f"{'Macro ROC-AUC (val)':<35} {auc_val:<18.4f} {auc_val:<18.4f}")
        print(f"{'Rare-species F1 (val)':<35} {f1_fixed_rare:<18.4f} {f1_cfar_rare:<18.4f}")
        print(f"{'FPR (val)':<35} {fpr_fixed_val:<18.4f} {fpr_cfar_val:<18.4f}")

        # ── CFAR threshold stats ───────────────────────────────────────
        print(f"\nCFAR threshold stats:")
        print(f"  min:  {t_cfar.min():.4f}")
        print(f"  max:  {t_cfar.max():.4f}")
        print(f"  mean: {t_cfar.mean():.4f}")
        print(f"  std:  {t_cfar.std():.4f}")

    # ── Soundscape FPR (if available) ──────────────────────────────
    sc_fpr_fixed = None
    sc_fpr_cfar = None
    if sc_probs is not None and sc_targets is not None:
        sc_preds_fixed = apply_threshold(sc_probs, t_fixed)
        sc_preds_cfar = apply_threshold(sc_probs, t_cfar)

        sc_fpr_fixed = false_positive_rate(sc_targets, sc_preds_fixed)
        sc_fpr_cfar = false_positive_rate(sc_targets, sc_preds_cfar)

        if verbose:
            print(f"\n{'FPR (soundscapes)':<35} {sc_fpr_fixed:<18.4f} {sc_fpr_cfar:<18.4f}")

    # ── Per-species threshold examples (top 10 highest / lowest) ──
    if verbose:
        sorted_idx = np.argsort(t_cfar)
        print(f"\n  Lowest CFAR thresholds (most sensitive):")
        for i in sorted_idx[:5]:
            print(f"    {labels[i]:<25} T={t_cfar[i]:.4f}")
        print(f"\n  Highest CFAR thresholds (most conservative):")
        for i in sorted_idx[-5:]:
            print(f"    {labels[i]:<25} T={t_cfar[i]:.4f}")

        print("\n" + "=" * 60)
        print("  Evaluation complete.")
        print("=" * 60)

    return {
        "k": float(k),
        "auc": float(auc_val),
        "f1_fixed": float(f1_fixed_rare),
        "f1_cfar": float(f1_cfar_rare),
        "fpr_fixed": float(fpr_fixed_val),
        "fpr_cfar": float(fpr_cfar_val),
        "fpr_sc_fixed": None if sc_fpr_fixed is None else float(sc_fpr_fixed),
        "fpr_sc_cfar": None if sc_fpr_cfar is None else float(sc_fpr_cfar),
        "threshold_mean": float(t_cfar.mean()),
        "threshold_std": float(t_cfar.std()),
    }


def save_k_sweep_plot(results: List[Dict[str, Any]], output_path: str = "k_sweep_figure.png") -> bool:
    """Save a dual-axis k-sensitivity plot to disk."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping figure generation.")
        return False

    ks = [r["k"] for r in results]
    f1s = [r["f1_cfar"] for r in results]
    fpr_val = [r["fpr_cfar"] for r in results]
    fpr_sc = [r["fpr_sc_cfar"] if r["fpr_sc_cfar"] is not None else np.nan for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ks, f1s, "o-", color="#1f77b4", linewidth=2, label="Rare-Species F1")
    ax1.set_xlabel("CFAR k")
    ax1.set_ylabel("Rare-Species F1", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ks, fpr_val, "s--", color="#d62728", linewidth=2, label="FPR (val)")
    if np.isfinite(fpr_sc).any():
        ax2.plot(ks, fpr_sc, "^--", color="#ff7f0e", linewidth=2, label="FPR (soundscapes)")
    ax2.set_ylabel("False Positive Rate", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [str(line.get_label()) for line in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Figure 1: FPR vs Rare-Species F1 Trade-off across k")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def run_k_sweep(
    backbone: str,
    k_values: List[float],
    fixed_t: float,
    batch_size: int,
    max_samples: int | None,
    include_soundscapes: bool,
) -> List[Dict[str, Any]]:
    """Run CFAR sensitivity sweep across multiple k values."""
    print("=" * 60)
    print("  BirdCLEF 2026 — CFAR k Sensitivity Sweep")
    print(f"  Backbone: {backbone} | k values: {k_values}")
    print("=" * 60)

    results: List[Dict[str, Any]] = []
    for k in k_values:
        print(f"\n--- Evaluating k={k} ---")
        metrics = evaluate(
            backbone=backbone,
            k=k,
            fixed_t=fixed_t,
            batch_size=batch_size,
            max_samples=max_samples,
            include_soundscapes=include_soundscapes,
            verbose=False,
        )
        if metrics is None:
            continue
        results.append(metrics)
        print(
            f"k={k:.2f} | Rare-F1={metrics['f1_cfar']:.4f} | "
            f"FPR(val)={metrics['fpr_cfar']:.4f} | "
            f"FPR(sc)={metrics['fpr_sc_cfar'] if metrics['fpr_sc_cfar'] is not None else float('nan'):.4f}"
        )

    if not results:
        print("No sweep results generated.")
        return results

    with open("k_sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved k_sweep_results.json")

    plotted = save_k_sweep_plot(results, output_path="k_sweep_figure.png")
    if plotted:
        print("Saved k_sweep_figure.png")

    print("\n" + "=" * 75)
    print("k      Rare-F1    FPR(val)   FPR(soundscapes)   AUC")
    print("-" * 75)
    for r in results:
        fpr_sc = r["fpr_sc_cfar"] if r["fpr_sc_cfar"] is not None else float("nan")
        print(f"{r['k']:<6.2f} {r['f1_cfar']:<10.4f} {r['fpr_cfar']:<10.4f} {fpr_sc:<18.4f} {r['auc']:<8.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 — Threshold Evaluation")
    parser.add_argument("--backbone", default="small",
                        choices=list(BACKBONE_BUILDERS.keys()))
    parser.add_argument("--k", type=float, default=2.0,
                        help="CFAR k multiplier (default: 2.0)")
    parser.add_argument("--fixed-t", type=float, default=0.5,
                        help="Fixed threshold value (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap total metadata rows (for quick eval)")
    parser.add_argument("--include-soundscapes", action="store_true",
                        help="Also evaluate on soundscape windows")
    parser.add_argument("--k-sweep", type=float, nargs="+", default=None,
                        help="Run sensitivity sweep for a list of k values (e.g. --k-sweep 1.0 1.5 2.0)")

    args = parser.parse_args()
    if args.k_sweep:
        run_k_sweep(
            backbone=args.backbone,
            k_values=args.k_sweep,
            fixed_t=args.fixed_t,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            include_soundscapes=args.include_soundscapes,
        )
    else:
        evaluate(
            backbone=args.backbone,
            k=args.k,
            fixed_t=args.fixed_t,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            include_soundscapes=args.include_soundscapes,
        )


if __name__ == "__main__":
    main()
