"""
Model training pipeline for pneumonia detection.

This module handles model compilation, training, evaluation, and saving
with comprehensive logging and callbacks.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, AUC, BinaryAccuracy, CategoricalAccuracy, Precision, Recall
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, Callback

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles model training with callbacks, validation, and evaluation.
    """
    
    def __init__(
        self,
        model: Model,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        loss_function: str = "binary_crossentropy",
        metrics: list[str] | None = None,
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            model: Keras model to train
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
            learning_rate: Initial learning rate
            loss_function: Loss function name
            metrics: List of metrics to track
        """
        self.model = model
        self.history = None
        self.metrics_dict = {}
        
        if metrics is None:
            metrics = ["accuracy"]
        
        self._compile_model(optimizer, learning_rate, loss_function, metrics)
    
    def _compile_model(
        self,
        optimizer: str,
        learning_rate: float,
        loss_function: str,
        metrics: list[str]
    ) -> None:
        """Compile the model with specified settings."""
        # Create optimizer
        if optimizer.lower() == "adam":
            try:
                opt = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
            except (AttributeError, ImportError):
                opt = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == "sgd":
            try:
                opt = keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9)
            except (AttributeError, ImportError):
                opt = SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == "rmsprop":
            try:
                opt = keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)
            except (AttributeError, ImportError):
                opt = RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Create loss function
        is_categorical = False
        if loss_function.lower() == "binary_crossentropy":
            loss = BinaryCrossentropy()
        elif loss_function.lower() == "categorical_crossentropy":
            loss = CategoricalCrossentropy()
            is_categorical = True
        else:
            loss = loss_function
            if "categorical" in str(loss_function).lower():
                is_categorical = True
        
        # Create metrics
        metric_objects = []
        for metric in metrics:
            if metric.lower() == "accuracy":
                if is_categorical:
                    metric_objects.append(CategoricalAccuracy(name="accuracy"))
                else:
                    metric_objects.append(BinaryAccuracy(name="accuracy"))
            elif metric.lower() == "categorical_accuracy":
                metric_objects.append(CategoricalAccuracy(name="categorical_accuracy"))
            elif metric.lower() == "binary_accuracy":
                metric_objects.append(BinaryAccuracy(name="binary_accuracy"))
            elif metric.lower() == "precision":
                metric_objects.append(Precision(name="precision"))
            elif metric.lower() == "recall":
                metric_objects.append(Recall(name="recall"))
            elif metric.lower() == "auc":
                metric_objects.append(AUC(name="auc"))
            else:
                metric_objects.append(metric)
        
        # Compile model
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metric_objects if metric_objects else ["accuracy"]
        )
        
        logger.info(f"Compiled model with {optimizer} optimizer")
    
    def train(
        self,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray] | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5,
        verbose: int = 1,
    ) -> dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val), optional
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            verbose: Verbosity level
            
        Returns:
            dict: Training history and metrics
        """
        X_train, y_train = train_data
        
        # Create callbacks
        callbacks = self._create_callbacks(
            early_stopping_patience=early_stopping_patience,
            reduce_lr_patience=reduce_lr_patience
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0 if verbose != 1 else 1
        )
        
        logger.info(f"Training completed for {len(self.history.history['loss'])} epochs")
        
        return self._extract_history()
    
    def train_generator(
        self,
        train_gen: Any,
        val_gen: Any | None = None,
        epochs: int = 50,
        class_weights: dict[int, float] | None = None,
        callbacks: list[Callback] | None = None,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5,
        verbose: int = 1,
    ) -> dict[str, Any]:
        """
        Train the model using data generators (memory-efficient streaming from disk).

        Args:
            train_gen: Training data generator
            val_gen: Validation data generator (optional)
            epochs: Number of training epochs
            class_weights: Optional dictionary of class weights
            callbacks: List of Keras callbacks (defaults to EarlyStopping & ReduceLROnPlateau)
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            verbose: Verbosity level

        Returns:
            dict: Training history
        """
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if val_gen is not None else 'loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if val_gen is not None else 'loss',
                    factor=0.5,
                    patience=reduce_lr_patience,
                    min_lr=1e-7,
                    verbose=1
                ),
            ]

        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=verbose
        )

        logger.info(f"Generator training completed for {len(self.history.history['loss'])} epochs")
        return self._extract_history()
    
    def _create_callbacks(
        self,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5
    ) -> list[Callback]:
        """Create training callbacks."""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if hasattr(self, 'val_data') else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if hasattr(self, 'val_data') else 'loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
        ]
        return callbacks
    
    def _extract_history(self) -> dict[str, Any]:
        """Extract training history as dictionary."""
        if self.history is None:
            return {}
        
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        return history_dict
    
    def evaluate(
        self,
        test_data: tuple[np.ndarray, np.ndarray],
        batch_size: int = 32
    ) -> dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Tuple of (X_test, y_test)
            batch_size: Batch size for evaluation
            
        Returns:
            dict: Evaluation metrics
        """
        X_test, y_test = test_data
        
        results = self.model.evaluate(X_test, y_test, batch_size=batch_size)
        
        # Extract metrics
        metric_names = self.model.metrics_names
        metrics_dict = {}
        
        if isinstance(results, (list, tuple)):
            for name, value in zip(metric_names, results):
                metrics_dict[name] = float(value)
        else:
            metrics_dict[metric_names[0]] = float(results)
        
        self.metrics_dict = metrics_dict
        logger.info(f"Evaluation metrics: {metrics_dict}")
        
        return metrics_dict
    
    def evaluate_generator(
        self,
        test_gen: Any,
        verbose: int = 1
    ) -> dict[str, float]:
        """
        Evaluate model on a test generator.
        
        Args:
            test_gen: Test data generator
            verbose: Verbosity level
            
        Returns:
            dict: Evaluation metrics
        """
        results = self.model.evaluate(test_gen, verbose=verbose)
        metric_names = self.model.metrics_names
        metrics_dict = {}
        
        if isinstance(results, (list, tuple)):
            for name, value in zip(metric_names, results):
                metrics_dict[name] = float(value)
        else:
            metrics_dict[metric_names[0]] = float(results)
        
        self.metrics_dict = metrics_dict
        logger.info(f"Evaluation metrics: {metrics_dict}")
        return metrics_dict
    
    def save_model(self, filepath: str, include_optimizer: bool = True) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
            include_optimizer: Whether to save optimizer state
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(filepath, include_optimizer=include_optimizer)
        logger.info(f"Model saved to {filepath}")
    
    def save_history(self, filepath: str) -> None:
        """
        Save training history to JSON.
        
        Args:
            filepath: Path to save history
        """
        if self.history is None:
            logger.warning("No training history to save")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        history_dict = self._extract_history()
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"Training history saved to {filepath}")
    
    def save_metrics(self, filepath: str) -> None:
        """
        Save evaluation metrics to JSON.
        
        Args:
            filepath: Path to save metrics
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics_dict, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")


class FineTuner:
    """
    Fine-tune pre-trained models for specific tasks.
    """
    
    def __init__(self, model: Model) -> None:
        """
        Initialize fine-tuner.
        
        Args:
            model: Pre-trained model
        """
        self.model = model
        self.initial_state = None
    
    def save_initial_state(self) -> None:
        """Save initial model state before fine-tuning."""
        self.initial_state = {
            layer.name: layer.trainable for layer in self.model.layers
        }
    
    def unfreeze_layers(self, num_layers: int = -1) -> None:
        """
        Unfreeze layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from end
                       (-1 to unfreeze all)
        """
        if num_layers == -1:
            num_layers = len(self.model.layers)
        
        for layer in self.model.layers[-num_layers:]:
            layer.trainable = True
        
        logger.info(f"Unfroze last {num_layers} layers")
    
    def freeze_layers(self, num_layers: int = -1) -> None:
        """
        Freeze layers to prevent training.
        
        Args:
            num_layers: Number of layers to freeze from start
                       (-1 to freeze all)
        """
        if num_layers == -1:
            num_layers = len(self.model.layers)
        
        for layer in self.model.layers[:num_layers]:
            layer.trainable = False
        
        logger.info(f"Froze first {num_layers} layers")
    
    def reset_to_initial_state(self) -> None:
        """Reset model layers to initial trainability state."""
        if self.initial_state is None:
            logger.warning("No initial state saved")
            return
        
        for layer in self.model.layers:
            if layer.name in self.initial_state:
                layer.trainable = self.initial_state[layer.name]
        
        logger.info("Reset model to initial state")


class LearningRateScheduler:
    """Schedule learning rate changes during training."""
    
    @staticmethod
    def exponential_decay(
        initial_lr: float,
        decay_rate: float,
        decay_steps: int
    ) -> keras.callbacks.LearningRateScheduler:
        """
        Create exponential decay learning rate scheduler.
        
        Args:
            initial_lr: Initial learning rate
            decay_rate: Decay rate per step
            decay_steps: Number of steps for decay
            
        Returns:
            keras.callbacks.LearningRateScheduler: Callback
        """
        def lr_schedule(epoch, lr):
            return initial_lr * (decay_rate ** (epoch / decay_steps))
        
        return keras.callbacks.LearningRateScheduler(lr_schedule)
    
    @staticmethod
    def step_decay(
        initial_lr: float,
        drop_rate: float,
        epochs_per_drop: int
    ) -> keras.callbacks.LearningRateScheduler:
        """
        Create step decay learning rate scheduler.
        
        Args:
            initial_lr: Initial learning rate
            drop_rate: Multiplication factor when dropping
            epochs_per_drop: Epochs between drops
            
        Returns:
            keras.callbacks.LearningRateScheduler: Callback
        """
        def lr_schedule(epoch, lr):
            return initial_lr * (drop_rate ** (epoch // epochs_per_drop))
        
        return keras.callbacks.LearningRateScheduler(lr_schedule)
