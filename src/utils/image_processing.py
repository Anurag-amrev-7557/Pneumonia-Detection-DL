"""
Image processing utilities for the pneumonia detection system.

This module handles image loading, preprocessing, normalization, and validation
for model training and inference.
"""

import logging
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Handles all image processing operations for the pneumonia detection system.
    
    Provides methods for loading, resizing, normalizing, and validating images
    in various formats and from different sources.
    """
    
    # Supported image formats
    SUPPORTED_FORMATS: ClassVar = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    
    # Standard normalization values for ImageNet
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    def __init__(self, target_size: tuple[int, int] = (224, 224)) -> None:
        """
        Initialize the image processor.
        
        Args:
            target_size: Target image dimensions (height, width)
        """
        self.target_size = target_size
        
    def load_image(self, image_path: str | Path) -> np.ndarray:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            np.ndarray: Loaded image as numpy array
            
        Raises:
            FileNotFoundError: If image file does not exist
            ValueError: If image format is not supported
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if image_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {image_path.suffix}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )
        
        # Load image in BGR format (OpenCV default)
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise TypeError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        logger.debug(f"Loaded image: {image_path} (shape: {image.shape})")
        return image
    
    def load_image_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """
        Validate and prepare an image from a numpy array.
        
        Args:
            image_array: Image as numpy array
            
        Returns:
            np.ndarray: Validated image array
            
        Raises:
            ValueError: If array is not a valid image
        """
        if not isinstance(image_array, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if len(image_array.shape) not in [2, 3]:
            raise ValueError(f"Invalid image dimensions: {image_array.shape}")
        
        # Convert grayscale to RGB if needed
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        # Ensure uint8 dtype
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        return image_array
    
    def resize_image(
        self, 
        image: np.ndarray, 
        size: tuple[int, int] | None = None,
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image array
            size: Target size (height, width). If None, uses self.target_size
            interpolation: OpenCV interpolation method
            
        Returns:
            np.ndarray: Resized image
        """
        if size is None:
            size = self.target_size
        
        resized = cv2.resize(image, (size[1], size[0]), interpolation=interpolation)
        logger.debug(f"Resized image to {size}")
        return resized
    
    def normalize_image(
        self, 
        image: np.ndarray,
        method: str = "imagenet"
    ) -> np.ndarray:
        """
        Normalize image pixel values.
        
        Args:
            image: Input image array (values 0-255)
            method: Normalization method ('imagenet', 'standard', '01')
                - 'imagenet': ImageNet mean/std normalization
                - 'standard': Mean 0, std 1 normalization
                - '01': Scale to [0, 1]
            
        Returns:
            np.ndarray: Normalized image
            
        Raises:
            ValueError: If method is not supported
        """
        # Convert to float32
        image = image.astype(np.float32)
        
        if method == "imagenet":
            # ImageNet normalization
            image = image / 255.0
            image = (image - self.IMAGENET_MEAN) / self.IMAGENET_STD
        elif method == "standard":
            # Standard normalization (0-1 then standardize)
            image = image / 255.0
            mean = image.mean(axis=(0, 1), keepdims=True)
            std = image.std(axis=(0, 1), keepdims=True)
            image = (image - mean) / (std + 1e-8)
        elif method == "01":
            # Simple 0-1 scaling
            image = image / 255.0
        else:
            raise ValueError(
                f"Unsupported normalization method: {method}. "
                f"Use 'imagenet', 'standard', or '01'"
            )
        
        logger.debug(f"Applied {method} normalization")
        return image
    
    def denormalize_image(
        self, 
        image: np.ndarray,
        method: str = "imagenet"
    ) -> np.ndarray:
        """
        Reverse normalization to get original pixel range.
        
        Args:
            image: Normalized image array
            method: Normalization method used
            
        Returns:
            np.ndarray: Denormalized image (0-255 range)
        """
        image = image.astype(np.float32)
        
        if method == "imagenet":
            image = (image * self.IMAGENET_STD) + self.IMAGENET_MEAN
            image = np.clip(image * 255.0, 0, 255)
        elif method == "01":
            image = np.clip(image * 255.0, 0, 255)
        else:
            # For 'standard' method, just clip to 0-255
            image = np.clip(image * 255.0, 0, 255)
        
        return image.astype(np.uint8)
    
    def preprocess_image(
        self,
        image: str | Path | np.ndarray,
        normalize: bool = True,
        normalization_method: str = "01"
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline for an image.
        
        Args:
            image: Image path or numpy array
            normalize: Whether to normalize pixel values
            normalization_method: Normalization method to use
            
        Returns:
            np.ndarray: Preprocessed image ready for model input
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        else:
            image = self.load_image_from_array(image)
        
        # Resize
        image = self.resize_image(image)
        
        # Normalize
        if normalize:
            image = self.normalize_image(image, method=normalization_method)
        
        logger.debug("Image preprocessing complete")
        return image
    
    def batch_preprocess_images(
        self,
        image_paths: list,
        normalize: bool = True,
        normalization_method: str = "imagenet"
    ) -> np.ndarray:
        """
        Preprocess multiple images at once.
        
        Args:
            image_paths: List of image paths or arrays
            normalize: Whether to normalize pixel values
            normalization_method: Normalization method to use
            
        Returns:
            np.ndarray: Batch of preprocessed images (batch_size, height, width, channels)
        """
        images = []
        for image_path in image_paths:
            try:
                img = self.preprocess_image(
                    image_path, 
                    normalize=normalize,
                    normalization_method=normalization_method
                )
                images.append(img)
            except (FileNotFoundError, ValueError, TypeError) as e:
                logger.warning(f"Failed to preprocess image {image_path}: {e}")
                continue
        
        if not images:
            raise ValueError("No images could be preprocessed from the batch")
        
        batch = np.array(images)
        logger.info(f"Preprocessed batch of {len(images)} images (shape: {batch.shape})")
        return batch
    
    def apply_histogram_equalization(
        self, 
        image: np.ndarray,
        method: str = "adaptive"
    ) -> np.ndarray:
        """
        Apply histogram equalization to improve contrast.
        
        Args:
            image: Input image array
            method: 'standard' or 'adaptive' (CLAHE)
            
        Returns:
            np.ndarray: Image with enhanced contrast
        """
        # Convert to LAB color space for better results
        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        else:
            lab = image
        
        if method == "adaptive":
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        else:
            # Standard histogram equalization
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        
        # Convert back to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            result = lab
        
        return result
    
    def validate_image(self, image_path: str | Path) -> bool:
        """
        Validate if a file is a readable image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            bool: True if valid image, False otherwise
        """
        try:
            image_path = Path(image_path)
            if image_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                return False
            image = cv2.imread(str(image_path))
            return image is not None
        except (FileNotFoundError, OSError, TypeError):
            return False
