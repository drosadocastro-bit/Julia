# Cibuco_Boriken - BirdCLEF+ 2026

Cibuco_Boriken combines Cibuco (river, Manati Puerto Rico) and Boriken (Taino name for Puerto Rico).

## About

BirdCLEF+ 2026 competition entry focused on biodiversity monitoring in the Brazilian Pantanal.

Primary method contribution:

- CFAR-inspired adaptive thresholding from radar signal processing, adapted for bioacoustic species detection.

## Architecture

- ARIA audio pipeline adapted for multilabel bird species classification
- 50% overlap windowing for soundscape inference
- Weighted BCE loss for rare-species imbalance
- CFAR adaptive thresholding for per-species decision calibration
- Backbones: SmallCNN, MobileNetV2, EfficientNet-B0, ResNet-18

## Quick Start

```bash
pip install -r requirements.txt
kaggle competitions download -c birdclef-2026
python -m birdclef.train --backbone small --epochs 10
python -m birdclef.evaluate_thresholds --k-sweep 1.0 1.5 2.0 2.5 3.0
```

## Colab

Use `birdclef_colab_cfar.ipynb` for GPU-backed training and CFAR k-sweep runs.

## Working Note

CFAR-Inspired Adaptive Thresholding for Bioacoustic Species Detection in the Brazilian Pantanal.

## Team

Danny (Cibuco_Boriken), FAA Air Traffic Systems Specialist, Bayamon Puerto Rico, with 20+ years of signal processing experience.

## License

MIT