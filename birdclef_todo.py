# BirdCLEF+ 2026 — Fix Backlog
# Reviewed by: Claude (Sonnet 4.6)
# Scaffold built by: Opus 4.6 + GitHub Copilot
# Date: March 11, 2026
# Priority: P1 (before first training run) → P2 (before first submission) → P3 (v2)
#
# Pass this file to Opus/Copilot with: 
# "Work through this TODO list top to bottom. 
#  The overlap fix implementation is in birdclef_overlap_fix.py"

# ══════════════════════════════════════════════════════
# P1 — BEFORE FIRST TRAINING RUN (do these first)
# ══════════════════════════════════════════════════════

TODO_P1 = [

    {
        "id": "P1-01",
        "file": "birdclef/config.py",
        "issue": "MIN_CONFIDENCE_TO_EMIT zeros out valid low-probability signals",
        "impact": "CRITICAL — hurts ROC-AUC on rare species",
        "fix": "Set MIN_CONFIDENCE_TO_EMIT = 0.0",
        "note": "ROC-AUC is threshold-free. Never zero probabilities in submission."
    },

    {
        "id": "P1-02",
        "file": "birdclef/config.py",
        "issue": "MAX_LABEL_COUNT = 250 is approximate",
        "impact": "LOW — but set exact after data download",
        "fix": "Set MAX_LABEL_COUNT = 234 after running: "
               "pd.read_csv('taxonomy.csv')['species_code'].nunique()",
        "note": "Verify exact count from taxonomy.csv, may differ from 234"
    },

    {
        "id": "P1-03",
        "file": "birdclef/features.py",
        "issue": "iter_soundscape_windows uses hard 5-sec cuts with zero overlap",
        "impact": "CRITICAL — birds at window boundaries missed entirely",
        "fix": "Replace iter_soundscape_windows with overlap version from "
               "birdclef_overlap_fix.py. Add pool_overlapping_predictions() function.",
        "note": "Use 50% overlap (hop_sec = window_sec / 2 = 2.5s)"
    },

    {
        "id": "P1-04",
        "file": "birdclef/inference.py",
        "issue": "Zeroing probs below MIN_CONFIDENCE_TO_EMIT destroys ranking signal",
        "impact": "CRITICAL — directly reduces ROC-AUC score",
        "fix": "Remove this line entirely: "
               "prob_values[prob_values < MIN_CONFIDENCE_TO_EMIT] = 0.0 "
               "Replace inner loop with overlap-aware version from birdclef_overlap_fix.py",
        "note": "Keep raw sigmoid outputs. Round to 6 decimal places (not 4) for ranking precision."
    },

    {
        "id": "P1-05",
        "file": "birdclef/inference.py",
        "issue": "iter_soundscape_windows only yields 2 values, inference unpacks 2",
        "impact": "WILL BREAK after P1-03 fix — overlap version yields 3 values",
        "fix": "Update unpacking from: "
               "for chunk, end_time in iter_soundscape_windows(...) "
               "to: "
               "for chunk, start_time, end_time in iter_soundscape_windows(...)",
        "note": "Apply AFTER P1-03. birdclef_overlap_fix.py has the full patched loop."
    },

    {
        "id": "P1-06",
        "file": "birdclef/model.py",
        "issue": "Dead code branch in BirdClassifier.load()",
        "impact": "LOW — cosmetic, but confusing",
        "fix": "Replace: "
               "if self.backbone_name == 'small': self.model = builder(num_species) "
               "else: self.model = builder(num_species) "
               "with: "
               "self.model = builder(num_species)",
        "note": "Both branches were identical — Opus left scaffolding behind"
    },

]


# ══════════════════════════════════════════════════════
# P2 — BEFORE FIRST KAGGLE SUBMISSION
# ══════════════════════════════════════════════════════

TODO_P2 = [

    {
        "id": "P2-01",
        "file": "birdclef/features.py",
        "issue": "Per-window normalization makes silent windows look identical to loud ones",
        "impact": "HIGH — background noise and real calls produce same input tensor",
        "fix": "Compute dataset-level mean and std during training. "
               "Save as norm_stats.json alongside model weights. "
               "Apply in audio_to_melspec: mel = (mel - dataset_mean) / dataset_std",
        "note": "Compute stats on training set only. Apply same stats at inference."
    },

    {
        "id": "P2-02",
        "file": "birdclef/features.py",
        "issue": "Classical features include Chroma + Tempo — ARIA genre DNA, wrong for birds",
        "impact": "MEDIUM — adds noise to ensemble path, wastes compute",
        "fix": "Remove: chroma_stft, beat_track(tempo) "
               "Add: spectral_flatness (tonal vs noise), "
               "     spectral_contrast (bird song vs background), "
               "     delta_mfcc (call dynamics over time)",
        "note": "Only matters if using classical feature ensemble path"
    },

    {
        "id": "P2-03",
        "file": "birdclef/model.py",
        "issue": "No Perch/BirdNET pretrained backbone option",
        "impact": "HIGH — ImageNet weights learn from cats/cars, Perch learned bird vocalizations",
        "fix": "Add build_perch() backbone using google-research/perch from HuggingFace: "
               "AutoModel.from_pretrained('google/bird-vocalization-classifier') "
               "Replace final head with nn.Linear(1024, num_species) "
               "Add 'perch' key to BACKBONE_BUILDERS dict",
        "note": "Perch is free, open source, allowed by competition rules. "
                "Should be DEFAULT backbone for serious training runs."
    },

    {
        "id": "P2-04",
        "file": "birdclef/config.py",
        "issue": "DEFAULT_THRESHOLD = 0.5 is too aggressive for sparse multilabel",
        "impact": "MEDIUM — used in get_top_species() human output, not submission",
        "fix": "Add separate thresholds: "
               "DISPLAY_THRESHOLD = 0.5   # for human-readable output only "
               "SUBMISSION_THRESHOLD = None  # never threshold submission probs",
        "note": "Submission should always use raw probabilities for ROC-AUC"
    },

    {
        "id": "P2-05",
        "file": "birdclef/train.py",
        "issue": "Unknown if training accounts for class imbalance (234 species, very unequal counts)",
        "impact": "HIGH — common species will dominate, rare species (where score is) ignored",
        "fix": "Add weighted BCE loss: "
               "compute class_weights = 1 / sqrt(species_count) from train_metadata.csv "
               "pass to nn.BCEWithLogitsLoss(pos_weight=class_weights)",
        "note": "Pantanal dataset likely has 10x more common waterbirds than rare mammals"
    },

]


# ══════════════════════════════════════════════════════
# P3 — V2 IMPROVEMENTS (after first scored submission)
# ══════════════════════════════════════════════════════

TODO_P3 = [

    {
        "id": "P3-01",
        "file": "birdclef/model.py (new)",
        "issue": "No Sound Event Detection (SED) — model ignores temporal position of calls",
        "impact": "HIGH for leaderboard — top teams use SED for precise localization",
        "fix": "Add SED head alongside classification head. "
               "Output: (num_species, time_frames) framewise predictions. "
               "Use attention pooling instead of global avg pool.",
        "note": "Reference: PSLA, PANNs, EfficientAT architectures"
    },

    {
        "id": "P3-02",
        "file": "birdclef/inference.py",
        "issue": "No test-time augmentation (TTA)",
        "impact": "MEDIUM — averaging augmented predictions reduces variance",
        "fix": "For each window, run 3-5 augmented versions: "
               "original, time_shift(±0.5s), add_gaussian_noise(snr=20dB) "
               "Average sigmoid outputs before building row",
        "note": "CPU cost: 3-5x inference time. Monitor 90-min limit."
    },

    {
        "id": "P3-03",
        "file": "birdclef/ (new file: ensemble.py)",
        "issue": "Single model submission — no ensemble",
        "impact": "HIGH — all top BirdCLEF solutions are ensembles",
        "fix": "Train 3 models: Perch backbone, EfficientNet-B0, MobileNetV2 "
               "Average predictions with learned weights "
               "Diversity > individual accuracy for ensemble gains",
        "note": "Only worth doing after P2 fixes are in place"
    },

    {
        "id": "P3-04",
        "file": "birdclef/ (new file: pseudo_label.py)",
        "issue": "Not using unlabeled soundscape data for training",
        "impact": "MEDIUM-HIGH — pseudo-labeling on test soundscapes is standard in BirdCLEF",
        "fix": "After first submission: "
               "Run inference on test_soundscapes with threshold=0.8 "
               "Add high-confidence predictions as pseudo-labels "
               "Retrain with mixed real + pseudo data",
        "note": "Risk: error amplification if base model is weak. Do P2 first."
    },

    {
        "id": "P3-05",
        "file": "Working Note",
        "issue": "Not documenting as we build = missing $2,500 bonus prize",
        "impact": "$$$ — top 2 working notes each get $2,500",
        "fix": "Document each major decision as you make it: "
               "architecture choices, ARIA lineage, Julia conservation framing, "
               "overlap fix rationale, Perch vs ImageNet comparison "
               "Title: 'Extending Julia: Agentic Agricultural AI for Conservation Bioacoustics'",
        "note": "Publish at CLEF 2026 conference. Deadline same as competition."
    },

]


# ══════════════════════════════════════════════════════
# QUICK REFERENCE — Files touched by these fixes
# ══════════════════════════════════════════════════════

FILES_MODIFIED = {
    "birdclef/config.py":    ["P1-01", "P1-02", "P2-04"],
    "birdclef/features.py":  ["P1-03", "P1-05", "P2-01", "P2-02"],
    "birdclef/inference.py": ["P1-04", "P1-05", "P3-02"],
    "birdclef/model.py":     ["P1-06", "P2-03", "P3-01"],
    "birdclef/train.py":     ["P2-05"],
    "birdclef/ensemble.py":  ["P3-03"],  # new file
    "birdclef/pseudo_label.py": ["P3-04"],  # new file
}

ESTIMATED_LEADERBOARD_IMPACT = {
    "P1 fixes only":      "Top 40% → Top 25%",
    "P1 + P2 fixes":      "Top 25% → Top 15%",
    "P1 + P2 + P3 fixes": "Top 15% → Top 10%",
    "Prize territory":    "Top 5 — needs domain expertise + luck",
}
