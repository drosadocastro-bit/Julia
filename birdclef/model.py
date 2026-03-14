"""
birdclef.model — CNN architecture for bird species classification.

Lineage: Adapted from Aria's genre_cnn.py (SmallCNN + MobileNetV2 backbone).
Key changes:
  - Multilabel output (sigmoid) instead of multiclass (softmax)
  - BCE loss instead of CrossEntropy
  - Designed for ≤90 min CPU inference on Kaggle
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from birdclef.config import MODEL_DIR, MODEL_FILENAME, LABELS_FILENAME

logger = logging.getLogger("birdclef.model")


# ═══════════════════════════════════════════════════════════════════
# Backbones
# ═══════════════════════════════════════════════════════════════════

class BirdSmallCNN(nn.Module):
    """
    Lightweight CNN for fast CPU inference.
    Input: (B, 3, 224, 224) mel-spectrogram image.
    Output: (B, num_species) raw logits.
    """

    def __init__(self, num_species: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_species),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_efficientnet_b0(num_species: int) -> nn.Module:
    """
    EfficientNet-B0 with custom head for multilabel bird classification.
    Good accuracy/speed balance for CPU inference.
    """
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # Replace final classifier
    in_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_species),
    )
    return base


def build_mobilenet_v2(num_species: int) -> nn.Module:
    """
    MobileNetV2 — same backbone available in Aria.
    Very fast on CPU.
    """
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    base = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    in_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_species),
    )
    return base


def build_perch(num_species: int) -> nn.Module:
    """
    "Perch" backbone — frozen pretrained encoder + trainable head.

    NOTE: The real Google Perch is on TF Hub (not HuggingFace Transformers).
    We use ResNet-18 with ImageNet weights as a practical substitute:
      - Pretrained features (better than random init SmallCNN)
      - Frozen encoder → only head trains (fast convergence, less overfit)
      - CPU-friendly (~11M params, half frozen)
      - Answers the key question: does pretrained > scratch?

    For true Perch, see: https://tfhub.dev/google/bird-vocalization-classifier
    """
    from torchvision.models import resnet18, ResNet18_Weights

    base = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all encoder layers — only train the head
    for param in base.parameters():
        param.requires_grad = False

    # Replace fc head with trainable classifier
    in_features = base.fc.in_features  # 512
    base.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_species),
    )
    return base


def build_efficientnet_b2(num_species: int) -> nn.Module:
    """
    EfficientNet-B2 — larger capacity than B0 for stronger representation.
    ~9M params, heavier than B0 but still feasible on CPU within 90 min.
    """
    from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

    base = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    in_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_species),
    )
    return base


def build_convnext_tiny(num_species: int) -> nn.Module:
    """
    ConvNeXt-Tiny — modern pure-ConvNet with strong ImageNet performance.
    ~28M params, competitive with ViT-Small on spectrograms.
    """
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

    base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = base.classifier[2].in_features
    base.classifier[2] = nn.Linear(in_features, num_species)
    return base


# ═══════════════════════════════════════════════════════════════════
# Classifier Wrapper
# ═══════════════════════════════════════════════════════════════════

BACKBONE_BUILDERS = {
    "small": BirdSmallCNN,
    "efficientnet_b0": build_efficientnet_b0,
    "efficientnet_b2": build_efficientnet_b2,
    "mobilenet_v2": build_mobilenet_v2,
    "convnext_tiny": build_convnext_tiny,
    "perch": build_perch,
}


class BirdClassifier:
    """
    Wraps model loading, prediction, and label management.

    Usage:
        clf = BirdClassifier(backbone="efficientnet_b0")
        clf.load()
        probs = clf.predict(mel_tensor)  # shape: (num_species,)
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        model_dir: Path = MODEL_DIR,
        device: Optional[str] = None,
        temperature: float = 1.0,
    ):
        self.backbone_name = backbone
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.model: Optional[nn.Module] = None
        self.labels: List[str] = []
        self.is_loaded = False

    def load(self, model_path: Optional[Path] = None, labels_path: Optional[Path] = None):
        """Load a trained model and its label list from disk."""
        model_path = model_path or (self.model_dir / MODEL_FILENAME)
        labels_path = labels_path or (self.model_dir / LABELS_FILENAME)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        num_species = len(self.labels)
        builder = BACKBONE_BUILDERS.get(self.backbone_name)
        if builder is None:
            raise ValueError(f"Unknown backbone: {self.backbone_name}. Options: {list(BACKBONE_BUILDERS.keys())}")

        self.model = builder(num_species)

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        logger.info(f"Loaded BirdClassifier ({self.backbone_name}, {num_species} species) on {self.device}")

    def predict(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        """
        Predict species probabilities for a single mel-spectrogram tensor.

        Args:
            mel_tensor: shape (3, 224, 224) — one window

        Returns:
            1-D tensor of shape (num_species,) with sigmoid probabilities
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        assert self.model is not None
        x = mel_tensor.unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits / self.temperature).squeeze(0).cpu()

        return probs

    def predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Predict on a batch of mel-spectrograms.

        Args:
            batch: shape (B, 3, 224, 224)

        Returns:
            tensor of shape (B, num_species) with sigmoid probabilities
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        assert self.model is not None
        batch = batch.to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.sigmoid(logits / self.temperature).cpu()

        return probs

    def get_top_species(
        self, probs: torch.Tensor, threshold: float = 0.5, top_k: int = 5,
    ) -> List[Dict[str, float]]:
        """Return sorted list of species above threshold."""
        results = []
        for idx in torch.argsort(probs, descending=True)[:top_k]:
            score = float(probs[idx])
            if score >= threshold:
                results.append({
                    "species": self.labels[idx],
                    "probability": round(score, 4),
                })
        return results
