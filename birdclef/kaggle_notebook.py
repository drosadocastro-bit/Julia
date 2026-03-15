"""
BirdCLEF+ 2026 — Kaggle Submission Notebook
============================================
Team Cibuco_Boriken | EfficientNet-B2 | Smart Crop + CFAR

This script is the Kaggle kernel entry point.
Copy/paste into a Kaggle notebook cell, or upload as a .py script.

Requirements:
  - Attach competition dataset: birdclef-2026
  - Attach trained model dataset (your uploaded B2 checkpoint)
  - CPU kernel, internet OFF

Expected runtime: < 60 minutes on Kaggle CPU
"""

# ── Cell 1: Install dependencies (if needed) ──────────────────────
# !pip install -q librosa torchvision torchaudio --no-deps

# ── Cell 2: Imports ────────────────────────────────────────────────
import sys
import os

# Add module root to path (if running from notebook)
MODULE_ROOT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "/kaggle/working"
if MODULE_ROOT not in sys.path:
    sys.path.insert(0, MODULE_ROOT)

from pathlib import Path

# ── Cell 3: Configuration ─────────────────────────────────────────
# If you uploaded your trained model as a Kaggle dataset, set the path here:
MODEL_DATASET = "/kaggle/input/birdclef-julia-model"  # <-- change to your dataset name

# Override model directory to point to your attached dataset
os.environ["BIRDCLEF_MODEL_DIR"] = MODEL_DATASET

# ── Cell 4: Run Inference ─────────────────────────────────────────
from birdclef.config import OUTPUT_DIR, MODEL_DIR
from birdclef.inference import run_inference

# Point to the correct model dir
model_dir = Path(MODEL_DATASET) if os.path.exists(MODEL_DATASET) else MODEL_DIR

submission = run_inference(
    backbone="efficientnet_b2",  # must match training backbone
    model_dir=model_dir,
    batch_size=16,               # tune down if OOM on CPU
)

# ── Cell 5: Save submission ───────────────────────────────────────
output_path = Path("/kaggle/working/submission.csv")
submission.to_csv(output_path, index=False)
print(f"Submission saved: {output_path}")
print(f"Shape: {submission.shape}")

# Sanity checks
assert output_path.exists(), "submission.csv not created!"
assert len(submission) > 0, "Empty submission!"
print("All checks passed. Ready to submit.")
