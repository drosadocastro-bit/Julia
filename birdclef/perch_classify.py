"""
birdclef.perch_classify — Train a lightweight classifier on Perch embeddings.

Takes precomputed Perch embeddings (1280-dim) and trains a small MLP head
for bird species classification. The trained head is saved as a PyTorch
model for use in the Kaggle inference notebook.

Usage on Colab:
    python -m birdclef.perch_classify \
        --embeddings-dir perch_embeddings \
        --epochs 50 \
        --output perch_head.pt

Architecture: Linear(1280 → 512) → ReLU → Dropout → Linear(512 → num_species)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("birdclef.perch_classify")


class PerchHead(nn.Module):
    """Lightweight MLP classifier on top of Perch embeddings."""

    def __init__(self, embedding_dim: int = 1280, num_species: int = 206):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_species),
        )

    def forward(self, x):
        return self.classifier(x)


def train_perch_head(
    embeddings_dir: str,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    output_path: str = "perch_head.pt",
    patience: int = 10,
) -> None:
    """Train PerchHead on precomputed embeddings."""
    embeddings_dir = Path(embeddings_dir)

    # Load data
    embeddings = np.load(embeddings_dir / "embeddings.npy")  # (N, 1280)
    labels = np.load(embeddings_dir / "labels.npy")          # (N, num_species)
    with open(embeddings_dir / "species.json") as f:
        species_list = json.load(f)

    num_species = len(species_list)
    logger.info(f"Loaded {len(embeddings)} embeddings, {num_species} species")

    # Train/val split (90/10)
    n = len(embeddings)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.9 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = torch.from_numpy(embeddings[train_idx])
    y_train = torch.from_numpy(labels[train_idx])
    X_val = torch.from_numpy(embeddings[val_idx])
    y_val = torch.from_numpy(labels[val_idx])

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PerchHead(embedding_dim=embeddings.shape[1], num_species=num_species)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    logger.info(f"Training PerchHead: {epochs} epochs, batch={batch_size}, lr={lr}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(val_ds)

        elapsed = time.time() - t0
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model + metadata
            torch.save({
                "state_dict": model.state_dict(),
                "num_species": num_species,
                "embedding_dim": embeddings.shape[1],
                "species": species_list,
            }, output_path)
            improved = f"  -> Saved best (val_loss={val_loss:.6f})"
        else:
            epochs_no_improve += 1

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Time: {elapsed:.1f}s"
        )
        if improved:
            logger.info(improved)

        if patience > 0 and epochs_no_improve >= patience:
            logger.info(f"  Early stopping: no improvement for {patience} epochs")
            break

    logger.info(f"Best val_loss: {best_val_loss:.6f}")
    logger.info(f"Model saved to {output_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train Perch classifier head")
    parser.add_argument("--embeddings-dir", type=str, required=True,
                        help="Directory with embeddings.npy, labels.npy, species.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output", type=str, default="perch_head.pt",
                        help="Output path for trained head")
    args = parser.parse_args()

    train_perch_head(
        embeddings_dir=args.embeddings_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
