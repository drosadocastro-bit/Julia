"""
k-parameter sweep for CFAR adaptive thresholding.

Runs evaluate_thresholds with k in {1.0, 1.5, 2.0, 2.5, 3.0}
on the same 50-sample toy model and plots FPR vs Rare-Species F1.

Usage:
    python birdclef/k_sweep.py
"""

import json
import subprocess
import re
import sys


def run_sweep():
    k_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    results = []

    for k in k_values:
        print(f"\n{'='*40}")
        print(f"  Running k={k}")
        print(f"{'='*40}")

        out = subprocess.run(
            [
                sys.executable, "-m", "birdclef.evaluate_thresholds",
                "--backbone", "small",
                "--max-samples", "50",
                "--include-soundscapes",
                "--k", str(k),
            ],
            capture_output=True, text=True,
        )
        text = out.stdout + out.stderr

        row = {"k": k}
        for line in text.split("\n"):
            if "Macro ROC-AUC (val)" in line:
                nums = re.findall(r"[\d.]+", line.split("Macro ROC-AUC (val)")[1])
                row["auc"] = float(nums[0]) if nums else 0
            elif "Rare-species F1 (val)" in line:
                nums = re.findall(r"[\d.]+", line.split("Rare-species F1 (val)")[1])
                row["f1_fixed"] = float(nums[0]) if len(nums) >= 1 else 0
                row["f1_cfar"] = float(nums[1]) if len(nums) >= 2 else 0
            elif "FPR (val)" in line and "soundscapes" not in line:
                nums = re.findall(r"[\d.]+", line.split("FPR (val)")[1])
                row["fpr_fixed"] = float(nums[0]) if len(nums) >= 1 else 0
                row["fpr_cfar"] = float(nums[1]) if len(nums) >= 2 else 0
            elif "FPR (soundscapes)" in line:
                nums = re.findall(r"[\d.]+", line.split("FPR (soundscapes)")[1])
                row["fpr_sc_fixed"] = float(nums[0]) if len(nums) >= 1 else 0
                row["fpr_sc_cfar"] = float(nums[1]) if len(nums) >= 2 else 0
            elif "mean:" in line and "CFAR" not in line:
                nums = re.findall(r"[\d.]+", line)
                row["t_mean"] = float(nums[0]) if nums else 0

        results.append(row)
        print(f"  AUC={row.get('auc', '?')}")
        print(f"  F1_fixed={row.get('f1_fixed', '?')}, F1_cfar={row.get('f1_cfar', '?')}")
        print(f"  FPR_val_fixed={row.get('fpr_fixed', '?')}, FPR_val_cfar={row.get('fpr_cfar', '?')}")
        print(f"  FPR_sc_fixed={row.get('fpr_sc_fixed', '?')}, FPR_sc_cfar={row.get('fpr_sc_cfar', '?')}")
        print(f"  T_mean={row.get('t_mean', '?')}")

    # Save raw results
    out_path = "birdclef/k_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 75)
    print("  CFAR k-Sensitivity Sweep — Summary")
    print("=" * 75)
    print(f"{'k':<6} {'AUC':<8} {'F1_fix':<8} {'F1_cfar':<8} {'FPR_fix':<8} {'FPR_cfar':<9} {'FPR_sc':<8} {'T_mean':<8}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['k']:<6.1f} "
            f"{r.get('auc', 0):<8.4f} "
            f"{r.get('f1_fixed', 0):<8.4f} "
            f"{r.get('f1_cfar', 0):<8.4f} "
            f"{r.get('fpr_fixed', 0):<8.4f} "
            f"{r.get('fpr_cfar', 0):<9.4f} "
            f"{r.get('fpr_sc_cfar', 0):<8.4f} "
            f"{r.get('t_mean', 0):<8.4f}"
        )

    return results


def plot_sweep(results):
    """
    Figure 1: FPR vs Rare-Species F1 trade-off curve for k sweep.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    ks = [r["k"] for r in results]
    f1_cfar = [r.get("f1_cfar", 0) for r in results]
    fpr_cfar = [r.get("fpr_cfar", 0) for r in results]
    fpr_sc = [r.get("fpr_sc_cfar", 0) for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Rare-species F1 on left y-axis
    color_f1 = "#2196F3"
    ax1.set_xlabel("CFAR k parameter", fontsize=12)
    ax1.set_ylabel("Rare-Species F1", color=color_f1, fontsize=12)
    line1 = ax1.plot(ks, f1_cfar, "o-", color=color_f1, linewidth=2,
                     markersize=8, label="Rare-Species F1")
    ax1.tick_params(axis="y", labelcolor=color_f1)
    ax1.set_ylim(0, 1.05)

    # FPR on right y-axis
    ax2 = ax1.twinx()
    color_fpr = "#F44336"
    color_sc = "#FF9800"
    ax2.set_ylabel("False Positive Rate", color=color_fpr, fontsize=12)
    line2 = ax2.plot(ks, fpr_cfar, "s--", color=color_fpr, linewidth=2,
                     markersize=8, label="FPR (val)")
    line3 = ax2.plot(ks, fpr_sc, "^--", color=color_sc, linewidth=2,
                     markersize=8, label="FPR (soundscapes)")
    ax2.tick_params(axis="y", labelcolor=color_fpr)
    ax2.set_ylim(0, max(max(fpr_cfar), max(fpr_sc)) * 1.3 + 0.01)

    # Annotate each point with k value
    for i, k in enumerate(ks):
        ax1.annotate(f"k={k}", (k, f1_cfar[i]),
                     textcoords="offset points", xytext=(0, 12),
                     fontsize=9, ha="center", color=color_f1)

    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=10)

    ax1.set_title(
        "Figure 1: CFAR k-Parameter Sensitivity\n"
        "FPR vs Rare-Species F1 Trade-off (50-sample toy model)",
        fontsize=13, fontweight="bold",
    )
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = "birdclef/k_sweep_figure.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved to {fig_path}")
    plt.close()


if __name__ == "__main__":
    results = run_sweep()
    plot_sweep(results)
