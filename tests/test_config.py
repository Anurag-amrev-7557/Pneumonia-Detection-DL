"""Tests for configuration management."""

import unittest

from src.config.settings import DataConfig, ModelConfig, Settings


class TestDataConfig(unittest.TestCase):
    """Test cases for DataConfig."""
    
    def test_valid_splits(self):
        """Test valid data splits."""
        config = DataConfig(train_split=0.7, val_split=0.15, test_split=0.15)
        self.assertEqual(config.train_split + config.val_split + config.test_split, 1.0)
    
    def test_invalid_splits(self):
        """Test invalid data splits."""
        with self.assertRaises(ValueError):
            DataConfig(train_split=0.5, val_split=0.3, test_split=0.3)
    
    def test_invalid_image_size(self):
        """Test invalid image size."""
        with self.assertRaises(ValueError):
            DataConfig(image_size=(0, 0))


class TestModelConfig(unittest.TestCase):
    """Test cases for ModelConfig."""
    
    def test_valid_dropout(self):
        """Test valid dropout rate."""
        config = ModelConfig(dropout_rate=0.5)
        self.assertEqual(config.dropout_rate, 0.5)
    
    def test_invalid_dropout(self):
        """Test invalid dropout rate."""
        with self.assertRaises(ValueError):
            ModelConfig(dropout_rate=1.5)
    
    def test_invalid_learning_rate(self):
        """Test invalid learning rate."""
        with self.assertRaises(ValueError):
            ModelConfig(learning_rate=-0.001)


class TestSettings(unittest.TestCase):
    """Test cases for main Settings class."""
    
    def test_initialization(self):
        """Test Settings initialization."""
        settings = Settings()
        
        self.assertIsNotNone(settings.data)
        self.assertIsNotNone(settings.model)
        self.assertIsNotNone(settings.inference)
        self.assertIsNotNone(settings.ui)
    
    def test_config_dict(self):
        """Test getting configuration as dictionary."""
        settings = Settings()
        config = settings.get_config_dict()
        
        self.assertIn('data', config)
        self.assertIn('model', config)
        self.assertIn('inference', config)


if __name__ == '__main__':
    unittest.main()
