"""Tests for image processing utilities."""

import unittest

import numpy as np

from src.utils.image_processing import ImageProcessor


class TestImageProcessor(unittest.TestCase):
    """Test cases for ImageProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor(target_size=(224, 224))
    
    def test_initialization(self):
        """Test ImageProcessor initialization."""
        self.assertEqual(self.processor.target_size, (224, 224))
    
    def test_load_image_from_array(self):
        """Test loading image from numpy array."""
        # Create dummy image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        result = self.processor.load_image_from_array(img_array)
        
        self.assertEqual(result.shape, (256, 256, 3))
        self.assertEqual(result.dtype, np.uint8)
    
    def test_resize_image(self):
        """Test image resizing."""
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        resized = self.processor.resize_image(img_array)
        
        self.assertEqual(resized.shape, (224, 224, 3))
    
    def test_normalize_imagenet(self):
        """Test ImageNet normalization."""
        img_array = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        normalized = self.processor.normalize_image(img_array, method='imagenet')
        
        self.assertEqual(normalized.shape, (224, 224, 3))
        self.assertTrue(normalized.min() >= -2 and normalized.max() <= 2)
    
    def test_normalize_01(self):
        """Test 0-1 normalization."""
        img_array = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        normalized = self.processor.normalize_image(img_array, method='01')
        
        self.assertTrue(normalized.min() >= 0 and normalized.max() <= 1)
    
    def test_invalid_normalization_method(self):
        """Test invalid normalization method raises error."""
        img_array = np.ones((224, 224, 3), dtype=np.uint8)
        
        with self.assertRaises(ValueError):
            self.processor.normalize_image(img_array, method='invalid')


if __name__ == '__main__':
    unittest.main()
