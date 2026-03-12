"""
birdclef.config — Competition constants and paths.

All BirdCLEF-specific configuration lives here.
Kaggle paths differ from local dev paths; we detect environment automatically.
"""

import os
from pathlib import Path

# ── Environment Detection ──────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/input")

# ── Paths ──────────────────────────────────────────────────────────
if IS_KAGGLE:
    # Kaggle kernel paths (read-only input, writable working dir)
    COMPETITION_DIR = Path("/kaggle/input/birdclef-2026")
    TRAIN_AUDIO_DIR = COMPETITION_DIR / "train_audio"
    TRAIN_SOUNDSCAPES_DIR = COMPETITION_DIR / "train_soundscapes"
    TRAIN_SOUNDSCAPES_LABELS_CSV = COMPETITION_DIR / "train_soundscapes_labels.csv"
    TEST_AUDIO_DIR = COMPETITION_DIR / "test_soundscapes"
    TRAIN_META_CSV = COMPETITION_DIR / "train_metadata.csv"
    TAXONOMY_CSV = COMPETITION_DIR / "taxonomy.csv"
    SAMPLE_SUBMISSION = COMPETITION_DIR / "sample_submission.csv"
    MODEL_DIR = Path("/kaggle/input")  # attach trained model as dataset
    OUTPUT_DIR = Path("/kaggle/working")
else:
    # Local development paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    COMPETITION_DIR = PROJECT_ROOT / "data" / "birdclef-2026" / "birdclef-2026"
    TRAIN_AUDIO_DIR = COMPETITION_DIR / "train_audio"
    TRAIN_SOUNDSCAPES_DIR = COMPETITION_DIR / "train_soundscapes"
    TRAIN_SOUNDSCAPES_LABELS_CSV = COMPETITION_DIR / "train_soundscapes_labels.csv"
    TEST_AUDIO_DIR = COMPETITION_DIR / "test_soundscapes"
    TRAIN_META_CSV = COMPETITION_DIR / "train_metadata.csv"
    TAXONOMY_CSV = COMPETITION_DIR / "taxonomy.csv"
    SAMPLE_SUBMISSION = COMPETITION_DIR / "sample_submission.csv"
    MODEL_DIR = PROJECT_ROOT / "birdclef" / "models"
    OUTPUT_DIR = PROJECT_ROOT / "birdclef" / "output"

# ── Audio Constants ────────────────────────────────────────────────
SAMPLE_RATE = 32000          # BirdCLEF standard SR
WINDOW_SECONDS = 5           # Each prediction covers 5 seconds
N_MELS = 128                 # Mel-spectrogram bins
N_FFT = 2048                 # FFT window
HOP_LENGTH = 512             # Hop for STFT
FMIN = 50                    # Min frequency (Hz) — cut low rumble
FMAX = 14000                 # Max frequency (Hz) — bird vocalizations

# ── Model Constants ────────────────────────────────────────────────
MAX_LABEL_COUNT = 234        # Exact count from taxonomy.csv species_code column
MODEL_FILENAME = "birdclef_model.pt"
LABELS_FILENAME = "birdclef_labels.json"

# ── Training Hyperparameters ───────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 30
TRAIN_SPLIT = 0.9
MIXUP_ALPHA = 0.4            # Mixup augmentation strength

# ── Inference Thresholds ───────────────────────────────────────────
# Conservative: better to predict 0 than hallucinate a species
DEFAULT_THRESHOLD = 0.5
# ROC-AUC is threshold-free — never zero out probabilities in submission
MIN_CONFIDENCE_TO_EMIT = 0.0

# ── Ensure output dirs exist locally ──────────────────────────────
if not IS_KAGGLE:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
