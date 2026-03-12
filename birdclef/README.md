# BirdCLEF+ 2026 — Julia Bioacoustics Module

> *Julia learns to listen to the Pantanal.*

---

## What This Is

A competition-ready module for [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026),
built on top of the Julia AI Crop Caretaker codebase and adapted from
[Project Aria](https://github.com/drosadocastro-bit/Project-Aria)'s audio classification stack.

**Goal:** Identify wildlife species from 5-second audio windows in the Pantanal wetlands (Brazil).

**Prize pool:** $50,000 · **Deadline:** June 3, 2026

---

## Migration Map: Aria → BirdCLEF

| Component | Project Aria (Source) | BirdCLEF Module (Adapted) | Status |
|-----------|----------------------|---------------------------|--------|
| **Mel-spectrogram extraction** | `genre_cnn.py` — `predict_audio()` | `features.py` — `audio_to_melspec()` | ✅ Ported |
| **Classical features (MFCC/spectral)** | `genre_classifier.py` — `LiveAudioAnalyzer` | `features.py` — `extract_classical_features()` | ✅ Ported |
| **SmallCNN backbone** | `genre_cnn.py` — `SmallCNN` | `model.py` — `BirdSmallCNN` | ✅ Ported |
| **MobileNetV2 backbone** | `genre_cnn.py` — `build_mobilenet_v2()` | `model.py` — `build_mobilenet_v2()` | ✅ Ported |
| **EfficientNet-B0** | *Not in Aria* | `model.py` — `build_efficientnet_b0()` | ✅ New |
| **Mixup augmentation** | *Not in Aria* | `train.py` — `mixup_batch()` | ✅ New |
| **Multilabel output (sigmoid + BCE)** | Multiclass (softmax + CE) | `model.py` — `BirdClassifier.predict()` | ✅ Changed |
| **Soundscape windowing** | *Not needed in Aria* | `features.py` — `iter_soundscape_windows()` | ✅ New |
| **Kaggle inference pipeline** | *Not applicable* | `inference.py` — `run_inference()` | ✅ New |
| **Model validation / drift** | `model_validator.py` | *Planned for Phase 2* | 🔲 TODO |
| **Listener profile bias** | `listener_profile.py` | *Not applicable* | ➖ Skipped |
| **DSP / EQ presets** | `audio_intelligence.py` | *Not applicable* | ➖ Skipped |

---

## Module Structure

```
birdclef/
├── __init__.py              # Module marker
├── config.py                # Paths, constants, hyperparameters
├── features.py              # Audio → mel-spectrogram + classical features
├── model.py                 # CNN architectures + BirdClassifier wrapper
├── train.py                 # Training pipeline (local GPU/CPU)
├── inference.py             # Kaggle-compatible submission generator
├── kaggle_notebook.py       # Copy-paste Kaggle notebook template
├── models/                  # Trained model checkpoints (.pt, .json)
├── output/                  # Generated submission.csv files
└── tests/
    └── test_birdclef.py     # 16 tests covering features, model, training
```

---

## Quick Start

### 1. Download Competition Data

```bash
kaggle competitions download -c birdclef-2026 -p data/birdclef-2026
unzip data/birdclef-2026/birdclef-2026.zip -d data/birdclef-2026/
```

### 2. Train Locally

```bash
# Fast iteration (10% data, small backbone)
python -m birdclef.train --backbone small --epochs 5 --fast

# Full training (EfficientNet-B0, all data, mixup)
python -m birdclef.train --backbone efficientnet_b0 --epochs 30
```

### 3. Run Inference Locally

```bash
python -m birdclef.inference
# → birdclef/output/submission.csv
```

### 4. Submit on Kaggle

1. Upload `birdclef/models/birdclef_model.pt` + `birdclef_labels.json` as a Kaggle dataset
2. Create a new notebook attached to the competition
3. Add your model dataset as input
4. Paste the contents of `kaggle_notebook.py` into a cell
5. Submit — should run in < 60 min on CPU

---

## Kaggle Constraints Checklist

| Constraint | How We Handle It |
|------------|-----------------|
| CPU only (≤ 90 min) | EfficientNet-B0 runs ~1-2 sec/window on CPU; batched inference |
| No internet | Model loaded from attached dataset; no API calls |
| `submission.csv` output | `inference.py` auto-generates with correct column order |
| Freely available pretrained weights | ImageNet weights used for backbone init (torchvision) |
| No GPU at submission | `torch.device("cpu")` fallback everywhere |

---

## Evaluation Metric

**Macro-averaged ROC-AUC** (skipping classes with no true positives in the test set).

This means:
- High recall matters — don't miss species that ARE present
- Low false-positive rate matters — don't hallucinate species that AREN'T there
- Rare species count equally to common ones (macro averaging)

---

## Roadmap

- [x] Phase 1: Scaffold module, port Aria features, build CNN architectures
- [x] Phase 1: Training pipeline with mixup + cosine LR
- [x] Phase 1: Kaggle-compatible inference + submission generator
- [x] Phase 1: 16 unit tests, all green
- [ ] Phase 2: Download data, run first training pass
- [ ] Phase 2: Add model validation / drift detection (from Aria)
- [ ] Phase 3: Experiment with SED (Sound Event Detection) architecture
- [ ] Phase 3: Add SpecAugment (time/frequency masking)
- [ ] Phase 4: Pseudo-labeling on test soundscapes
- [ ] Phase 4: Ensemble (EfficientNet + MobileNet + classical RF)
- [ ] Phase 5: Final submission tuning + threshold optimization

---

## Key Differences from Aria's Audio Pipeline

| Aspect | Project Aria | BirdCLEF |
|--------|-------------|----------|
| **Task** | Music genre (10 classes, single-label) | Wildlife species (250+ classes, multilabel) |
| **Audio length** | 30-sec tracks | 5-sec windows from hours-long soundscapes |
| **Sample rate** | 22,050 Hz | 32,000 Hz (captures higher-pitched bird calls) |
| **Output** | Softmax (one genre per track) | Sigmoid (multiple species per window) |
| **Loss** | CrossEntropy | BCEWithLogitsLoss |
| **Freq range** | Full spectrum | 50–14,000 Hz (bird vocalization range) |
| **Deployment** | Real-time on Windows PC | Kaggle CPU notebook (offline) |

---

*Built with Julia's philosophy: Truth > Fluency, Evidence > Intuition.*
