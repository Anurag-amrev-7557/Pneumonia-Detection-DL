"""
Pneumonia Detection System

A professional deep learning system for automated pneumonia detection
from chest X-ray images using state-of-the-art CNN architectures.
"""

__version__ = "1.0.0"
__author__ = "Anurag"
__email__ = "your.email@example.com"

from src.config.settings import settings
from src.models.architectures import ModelFactory
from src.models.inference import PneumoniaDetector
from src.utils.image_processing import ImageProcessor

__all__ = [
    'ImageProcessor',
    'ModelFactory',
    'PneumoniaDetector',
    'settings',
]
