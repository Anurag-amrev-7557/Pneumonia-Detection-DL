"""
Hyperparameter tuning for model optimization.

Uses Keras Tuner for efficient hyperparameter search and optimization.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import keras_tuner as kt
import numpy as np
from tensorflow import keras

logger = logging.getLogger(__name__)


class PneumoniaDetectorTuner:
    """
    Hyperparameter tuner for pneumonia detection models.
    
    Uses Keras Tuner to find optimal hyperparameters through
    systematic search.
    """
    
    def __init__(
        self,
        project_name: str = "pneumonia_tuner",
        tuner_type: str = "bayesian",
        max_trials: int = 10,
        executions_per_trial: int = 1
    ) -> None:
        """
        Initialize hyperparameter tuner.
        
        Args:
            project_name: Name for tuner project
            tuner_type: Type of tuner ('bayesian', 'random', 'grid', 'hyperband')
            max_trials: Maximum number of trials
            executions_per_trial: Executions per trial for stability
        """
        self.project_name = project_name
        self.tuner_type = tuner_type
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.tuner = None
        self.best_model = None
        self.best_hps = None
    
    def build_model(self, hp: kt.HyperParameters) -> keras.Model:
        """
        Build model with hyperparameters to tune.
        
        Args:
            hp: HyperParameters object from Keras Tuner
            
        Returns:
            keras.Model: Built model
        """
        model = keras.Sequential()
        
        # Input Layer
        model.add(keras.layers.Input(shape=(224, 224, 3)))
        
        # First Conv Block
        model.add(keras.layers.Conv2D(
            filters=hp.Int('conv_1_filters', min_value=32, max_value=128, step=32),
            kernel_size=3,
            activation='relu',
            padding='same'
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D((2, 2)))
        
        # Second Conv Block
        model.add(keras.layers.Conv2D(
            filters=hp.Int('conv_2_filters', min_value=64, max_value=256, step=64),
            kernel_size=3,
            activation='relu',
            padding='same'
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Dropout(
            hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)
        ))
        
        # Third Conv Block (optional)
        if hp.Boolean('use_conv_3'):
            model.add(keras.layers.Conv2D(
                filters=hp.Int('conv_3_filters', min_value=128, max_value=256, step=64),
                kernel_size=3,
                activation='relu',
                padding='same'
            ))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.MaxPooling2D((2, 2)))
            model.add(keras.layers.Dropout(
                hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)
            ))
        
        # Global Pooling
        model.add(keras.layers.GlobalAveragePooling2D())
        
        # Dense Layers
        model.add(keras.layers.Dense(
            units=hp.Int('dense_1_units', min_value=128, max_value=512, step=128),
            activation='relu'
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(
            hp.Float('dropout_3', min_value=0.3, max_value=0.6, step=0.1)
        ))
        
        model.add(keras.layers.Dense(2, activation='softmax'))
        
        # Compile
        optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])
        learning_rate = hp.Float(
            'learning_rate',
            min_value=1e-4,
            max_value=1e-2,
            sampling='log'
        )
        
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def search(
        self,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray],
        epochs: int = 20,
        batch_size: int = 32,
        callbacks: list | None = None
    ) -> dict[str, Any]:
        """
        Perform hyperparameter search.
        
        Args:
            train_data: Training data (X, y)
            val_data: Validation data (X, y)
            epochs: Epochs per trial
            batch_size: Batch size
            callbacks: Additional callbacks
            
        Returns:
            dict: Search results including best hyperparameters
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Create tuner
        if self.tuner_type == 'bayesian':
            tuner_class = kt.BayesianOptimization
        elif self.tuner_type == 'random':
            tuner_class = kt.RandomSearch
        elif self.tuner_type == 'grid':
            tuner_class = kt.GridSearch
        elif self.tuner_type == 'hyperband':
            tuner_class = kt.Hyperband
        else:
            raise ValueError(f"Unknown tuner type: {self.tuner_type}")
        
        self.tuner = tuner_class(
            hypermodel=self.build_model,
            objective='val_accuracy',
            max_trials=self.max_trials,
            executions_per_trial=self.executions_per_trial,
            directory='tuning_results',
            project_name=self.project_name
        )
        
        # Setup callbacks
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        
        # Run search
        logger.info(f"Starting hyperparameter search with {self.tuner_type} tuner")
        self.tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Get best hyperparameters
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info(f"Best hyperparameters: {self.best_hps.values}")
        
        # Build and return best model
        self.best_model = self.tuner.hypermodel.build(self.best_hps)
        
        return {
            'best_hyperparameters': self.best_hps.values,
            'best_model': self.best_model,
            'search_results': self.tuner.results_summary()
        }
    
    def get_best_model(self) -> keras.Model:
        """Get the best model found during search."""
        if self.best_model is None:
            raise ValueError("No search has been performed yet")
        return self.best_model
    
    def save_best_model(self, filepath: str) -> None:
        """Save the best model."""
        if self.best_model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.best_model.save(filepath)
        logger.info(f"Saved best model to {filepath}")
    
    def save_search_results(self, filepath: str) -> None:
        """Save search results summary."""
        if self.tuner is None:
            raise ValueError("No search has been performed")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.tuner.results_summary(num_trials=self.max_trials)
        with open(filepath, 'w') as f:
            f.write(summary)
        
        logger.info(f"Saved search results to {filepath}")


class ManualHyperparameterTuner:
    """
    Manual hyperparameter tuning with grid search.
    
    Allows manual specification of hyperparameter ranges and
    systematic evaluation.
    """
    
    def __init__(self, model_builder: Callable) -> None:
        """
        Initialize manual tuner.
        
        Args:
            model_builder: Function that builds model given hyperparameters
        """
        self.model_builder = model_builder
        self.results = []
    
    def grid_search(
        self,
        param_grid: dict[str, list],
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray],
        epochs: int = 10,
        batch_size: int = 32
    ) -> dict[str, Any]:
        """
        Perform grid search over parameter combinations.
        
        Args:
            param_grid: Dictionary of parameter names and value lists
            train_data: Training data
            val_data: Validation data
            epochs: Epochs per model
            batch_size: Batch size
            
        Returns:
            dict: Best parameters and scores
        """
        import itertools
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"Starting grid search with {len(combinations)} combinations")
        
        best_score = 0
        best_params = None
        
        for i, values in enumerate(combinations):
            params = dict(zip(param_names, values))
            logger.info(f"Trial {i+1}/{len(combinations)}: {params}")
            
            try:
                # Build and train model
                model = self.model_builder(params)
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
                
                # Get validation score
                val_score = max(history.history['val_accuracy'])
                
                result = {
                    'params': params,
                    'val_accuracy': val_score,
                    'history': history.history
                }
                self.results.append(result)
                
                logger.info(f"  Val Accuracy: {val_score:.4f}")
                
                if val_score > best_score:
                    best_score = val_score
                    best_params = params
                
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error(f"  Failed: {e}")
                continue
        
        logger.info(f"Grid search complete. Best score: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.results
        }
    
    def get_results_dataframe(self):
        """Convert results to pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame([
                {**r['params'], 'val_accuracy': r['val_accuracy']}
                for r in self.results
            ])
        except ImportError:
            logger.warning("pandas not available for DataFrame conversion")
            return None
