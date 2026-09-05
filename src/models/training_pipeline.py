#!/usr/bin/env python3
"""
Pneumonia Detection - Model Training Pipeline

Trains deep learning models (ResNet50, VGG16, Custom CNN, InceptionV3)
using memory-efficient streaming generators directly from the stratified data directory.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config.settings import settings
from src.models.architectures import ModelFactory
from src.models.training import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train(
    data_dir: Path = Path("data"),
    model_type: str = "resnet50",
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    output_dir: Path = Path("models/current"),
    patience: int = 6,
) -> dict:
    """
    Run end-to-end model training and evaluation pipeline.

    Args:
        data_dir: Root dataset directory containing train/, val/, test/
        model_type: Model architecture to train
        epochs: Maximum number of epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        output_dir: Directory where checkpoints and metrics are saved
        patience: Early stopping patience

    Returns:
        dict: Evaluation metrics on the test set
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    for d in (train_dir, val_dir, test_dir):
        if not d.exists():
            raise FileNotFoundError(f"Required split directory not found: {d}")

    logger.info(f"Setting up data generators from {data_dir}...")

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    target_size = settings.data.image_size

    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=settings.data.random_seed,
    )

    val_generator = test_datagen.flow_from_directory(
        str(val_dir),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_generator = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    # Compute balanced class weights
    classes = np.unique(train_generator.classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_generator.classes,
    )
    class_weights = dict(enumerate(weights))
    logger.info(f"Computed class weights: {class_weights}")

    # Build model
    logger.info(f"Building {model_type} architecture...")
    model = ModelFactory.create(
        model_type=model_type,
        input_shape=(target_size[0], target_size[1], 3),
        num_classes=2,
        dropout_rate=settings.model.dropout_rate,
    )

    trainer = ModelTrainer(
        model=model,
        optimizer="adam",
        learning_rate=learning_rate,
        loss_function="categorical_crossentropy",
        metrics=["accuracy", "precision", "recall", "auc"],
    )

    checkpoint_path = output_dir / "best_model.h5"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(log_dir=str(log_dir)),
    ]

    logger.info(f"Starting training for up to {epochs} epochs...")
    history = trainer.train_generator(
        train_gen=train_generator,
        val_gen=val_generator,
        epochs=epochs,
        class_weights=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training history
    history_file = output_dir / "training_history.json"
    trainer.save_history(str(history_file))

    # Evaluate best model on test set
    logger.info("Evaluating best model on held-out test set...")
    test_metrics = trainer.evaluate_generator(test_generator)

    metadata = {
        "model_type": model_type,
        "epochs_trained": len(history.get("loss", [])),
        "training_date": datetime.now(timezone.utc).isoformat(),
        "test_metrics": test_metrics,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        },
        "dataset": {
            "training_samples": train_generator.samples,
            "validation_samples": val_generator.samples,
            "test_samples": test_generator.samples,
        },
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING & EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Model saved to:       {checkpoint_path}")
    for k, v in test_metrics.items():
        print(f"Test {k:15s}: {v:.4f}")
    print("=" * 60 + "\n")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train deep learning model for pneumonia detection")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset directory")
    parser.add_argument(
        "--model",
        default="resnet50",
        choices=["resnet50", "vgg16", "custom_cnn", "inceptionv3"],
        help="Model architecture",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--output-dir", type=Path, default=Path("models/current"), help="Output directory")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
