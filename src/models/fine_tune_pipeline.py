#!/usr/bin/env python3
"""
Fine-tuning script for ResNet50 pneumonia detection.
Loads best_model.h5, unfreezes the top 30 layers of ResNet50, and trains at lr=1e-5.
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fine_tune(
    model_path: str | Path = 'models/current/best_model.h5',
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    output_dir: str | Path = 'models/current'
) -> bool:
    """
    Fine-tune ResNet50 for pneumonia detection.
    
    Args:
        model_path: Path to best model checkpoint
        epochs: Number of fine-tuning epochs
        batch_size: Batch size
        learning_rate: Fine-tuning learning rate
        output_dir: Output directory for weights
    """
    try:
        logger.info("=" * 70)
        logger.info("🔬 PNEUMONIA DETECTION - RESNET50 FINE-TUNING")
        logger.info("=" * 70)
        
        image_size = settings.data.image_size
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load best model checkpoint
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"❌ Model checkpoint not found at {model_path}. Train the base model first!")
            return False
            
        logger.info(f"📂 Loading best checkpoint from {model_path}...")
        model = keras.models.load_model(str(model_path))
        
        # Unfreeze the base ResNet50 backbone (top 30 layers)
        logger.info("🔓 Unfreezing top 30 layers of ResNet50 backbone...")
        base_model = None
        for layer in model.layers:
            if layer.name == 'resnet50':
                base_model = layer
                break
                
        if base_model:
            base_model.trainable = True
            for layer in base_model.layers[:-30]:
                layer.trainable = False
                
        trainable_count = sum(keras.backend.count_params(p) for p in model.trainable_weights)
        non_trainable_count = sum(keras.backend.count_params(p) for p in model.non_trainable_weights)
        logger.info(f"✅ Trainable parameters: {trainable_count:,}")
        logger.info(f"✅ Non-trainable parameters: {non_trainable_count:,}")
        
        # Setup data generators
        logger.info("\n📂 Setting up data generators...")
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes={'NORMAL': 0, 'PNEUMONIA': 1},
            shuffle=True
        )
        
        val_generator = test_datagen.flow_from_directory(
            'data/val',
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes={'NORMAL': 0, 'PNEUMONIA': 1},
            shuffle=False
        )
        
        test_generator = test_datagen.flow_from_directory(
            'data/test',
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes={'NORMAL': 0, 'PNEUMONIA': 1},
            shuffle=False
        )
        
        # Compute balanced class weights
        train_labels = train_generator.classes
        weights_array = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = dict(enumerate(weights_array))
        logger.info(f"⚖️ Applied Class Weights: {class_weights}")
        
        # Compile model with a very low learning rate
        logger.info(f"\n⚙️ Compiling model for fine-tuning (LR={learning_rate})...")
        try:
            optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        except (AttributeError, ImportError):
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=6,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(output_dir / 'finetuned_best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(output_dir / 'finetune_logs'),
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        # Fine-tune training
        logger.info(f"\n🚀 Starting fine-tuning for {epochs} epochs (batch_size={batch_size})...")
        logger.info("=" * 70)
        
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        logger.info("=" * 70)
        logger.info("\n✅ Fine-tuning completed!")
        
        # Evaluate on test set
        logger.info("\n📈 Evaluating fine-tuned model on test set...")
        results = model.evaluate(test_generator, verbose=0)
        test_loss, test_acc, test_precision, test_recall, test_auc = results
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 FINAL FINE-TUNED TEST RESULTS:")
        logger.info("=" * 70)
        logger.info(f"  Accuracy:  {test_acc:.4f} (🎯 Target: >0.90)")
        logger.info(f"  Precision: {test_precision:.4f}")
        logger.info(f"  Recall:    {test_recall:.4f}")
        logger.info(f"  AUC:       {test_auc:.4f}")
        logger.info(f"  Loss:      {test_loss:.4f}")
        logger.info("=" * 70)
        
        # Save final fine-tuned model
        final_model_name = f"pneumonia_finetuned_resnet50_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.h5"
        final_model_path = output_dir / final_model_name
        model.save(str(final_model_path))
        logger.info(f"\n💾 Fine-tuned model saved: {final_model_path}")
        
        return True
        
    except Exception as e:
        logger.exception(f"❌ Fine-tuning failed: {e}")
        return False


def main():
    """Main CLI entry point for fine_tune.py."""
    parser = argparse.ArgumentParser(description='Fine-tune ResNet50 for pneumonia detection')
    parser.add_argument('--epochs', type=int, default=15, help='Number of fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Fine-tuning learning rate')
    parser.add_argument('--model-path', default='models/current/best_model.h5', help='Path to model checkpoint')
    parser.add_argument('--output-dir', default='models/current', help='Output directory')
    
    args = parser.parse_args()
    return fine_tune(
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)