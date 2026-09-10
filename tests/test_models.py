"""Tests for model architectures and training."""

import numpy as np

import pytest
from tensorflow import keras

from src.models.architectures import CustomCNNArchitecture, ModelFactory
from src.models.training import FineTuner, ModelTrainer


class TestModelFactory:
    """Test model factory creation."""
    
    @pytest.mark.parametrize("model_type", [
        'custom_cnn', 'vgg16', 'resnet50', 'inceptionv3'
    ])
    def test_create_model(self, model_type):
        """Test model creation for all architectures."""
        model = ModelFactory.create(
            model_type=model_type,
            input_shape=(224, 224, 3),
            num_classes=2
        )
        
        assert model is not None
        assert isinstance(model, keras.Model)
        assert model.output_shape == (None, 2)
    
    def test_unsupported_model_type(self):
        """Test error handling for unsupported model types."""
        with pytest.raises(ValueError):
            ModelFactory.create(
                model_type='invalid_model',
                input_shape=(224, 224, 3)
            )
    
    def test_get_supported_models(self):
        """Test supported models list."""
        models = ModelFactory.get_supported_models()
        assert 'custom_cnn' in models
        assert 'vgg16' in models
        assert 'resnet50' in models
        assert 'inceptionv3' in models


class TestCustomCNNArchitecture:
    """Test custom CNN architecture."""
    
    def test_build_model(self):
        """Test building custom CNN."""
        arch = CustomCNNArchitecture(
            input_shape=(224, 224, 3),
            num_classes=2,
            dropout_rate=0.5
        )
        
        model = arch.build()
        
        assert model is not None
        assert model.output_shape == (None, 2)
        assert len(model.layers) > 10
    
    def test_dropout_rate_validation(self):
        """Test dropout rate validation."""
        with pytest.raises(ValueError):
            CustomCNNArchitecture(dropout_rate=1.5)
    
    def test_input_shape_validation(self):
        """Test input shape validation."""
        with pytest.raises(ValueError):
            CustomCNNArchitecture(input_shape=(0, 0, 3))


class TestModelTrainer:
    """Test model training."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple test model."""
        return keras.Sequential([
            keras.layers.Dense(10, activation='relu', input_shape=(10,)),
            keras.layers.Dense(2, activation='softmax')
        ])
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy training data."""
        X_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.eye(2)[np.random.randint(0, 2, 100)]
        
        X_val = np.random.randn(20, 10).astype(np.float32)
        y_val = np.eye(2)[np.random.randint(0, 2, 20)]
        
        return (X_train, y_train), (X_val, y_val)
    
    def test_trainer_compilation(self, simple_model):
        """Test model compilation."""
        trainer = ModelTrainer(
            simple_model,
            optimizer='adam',
            learning_rate=0.001
        )
        
        assert trainer.model is not None
        assert trainer.model.optimizer is not None
    
    def test_trainer_invalid_optimizer(self, simple_model):
        """Test error handling for invalid optimizer."""
        with pytest.raises(ValueError):
            ModelTrainer(
                simple_model,
                optimizer='invalid_optimizer'
            )
    
    def test_trainer_train(self, simple_model, dummy_data):
        """Test training loop."""
        trainer = ModelTrainer(simple_model)
        
        history = trainer.train(
            train_data=dummy_data[0],
            val_data=dummy_data[1],
            epochs=2,
            batch_size=32,
            verbose=0
        )
        
        assert 'loss' in history
        assert len(history['loss']) == 2


class TestFineTuner:
    """Test fine-tuning functionality."""
    
    @pytest.fixture
    def test_model(self):
        """Create test model."""
        return keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(10,), name='layer1'),
            keras.layers.Dense(32, activation='relu', name='layer2'),
            keras.layers.Dense(2, activation='softmax', name='output')
        ])
    
    def test_unfreeze_layers(self, test_model):
        """Test layer unfreezing."""
        fine_tuner = FineTuner(test_model)
        
        # Freeze all
        for layer in test_model.layers:
            layer.trainable = False
        
        # Unfreeze last 1
        fine_tuner.unfreeze_layers(1)
        
        assert not test_model.layers[0].trainable
        assert not test_model.layers[1].trainable
        assert test_model.layers[2].trainable
    
    def test_save_and_reset_state(self, test_model):
        """Test saving and resetting model state."""
        fine_tuner = FineTuner(test_model)
        
        # Save initial state
        fine_tuner.save_initial_state()
        
        # Modify
        for layer in test_model.layers:
            layer.trainable = False
        
        # Reset
        fine_tuner.reset_to_initial_state()
        
        assert fine_tuner.initial_state is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
