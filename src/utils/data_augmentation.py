"""
Data augmentation utilities for training robust models.

This module provides various data augmentation techniques to increase dataset
diversity and improve model generalization.
"""

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AugmentationBase(ABC):
    """Abstract base class for augmentation techniques."""
    
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to image."""


class RandomRotation(AugmentationBase):
    """Random rotation augmentation."""
    
    def __init__(self, angle_range: int = 20) -> None:
        """
        Initialize rotation augmentation.
        
        Args:
            angle_range: Maximum rotation angle in degrees
        """
        self.angle_range = angle_range
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random rotation.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Rotated image
        """
        angle = np.random.uniform(-self.angle_range, self.angle_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return rotated


class RandomShift(AugmentationBase):
    """Random translation/shift augmentation."""
    
    def __init__(self, shift_range: float = 0.2) -> None:
        """
        Initialize shift augmentation.
        
        Args:
            shift_range: Maximum shift as fraction of image dimensions
        """
        self.shift_range = shift_range
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random translation.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Shifted image
        """
        h, w = image.shape[:2]
        shift_h = int(np.random.uniform(-self.shift_range, self.shift_range) * h)
        shift_w = int(np.random.uniform(-self.shift_range, self.shift_range) * w)
        
        M = np.float32([[1, 0, shift_w], [0, 1, shift_h]])
        shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return shifted


class RandomZoom(AugmentationBase):
    """Random zoom augmentation."""
    
    def __init__(self, zoom_range: float = 0.2) -> None:
        """
        Initialize zoom augmentation.
        
        Args:
            zoom_range: Maximum zoom factor
        """
        self.zoom_range = zoom_range
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random zoom.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Zoomed image
        """
        h, w = image.shape[:2]
        zoom = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
        
        new_h, new_w = int(h * zoom), int(w * zoom)
        zoomed = cv2.resize(image, (new_w, new_h))
        
        if zoom > 1:
            # Crop from center
            y = (new_h - h) // 2
            x = (new_w - w) // 2
            zoomed = zoomed[y:y+h, x:x+w]
        else:
            # Pad with reflection
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            zoomed = cv2.copyMakeBorder(
                zoomed, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w,
                cv2.BORDER_REFLECT
            )
        
        return zoomed


class RandomFlip(AugmentationBase):
    """Random flip augmentation."""
    
    def __init__(self, horizontal: bool = True, vertical: bool = False) -> None:
        """
        Initialize flip augmentation.
        
        Args:
            horizontal: Enable horizontal flipping
            vertical: Enable vertical flipping
        """
        self.horizontal = horizontal
        self.vertical = vertical
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random flip.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Flipped image
        """
        if self.horizontal and np.random.rand() > 0.5:
            image = cv2.flip(image, 1)  # Flip horizontally
        
        if self.vertical and np.random.rand() > 0.5:
            image = cv2.flip(image, 0)  # Flip vertically
        
        return image


class RandomShear(AugmentationBase):
    """Random shear transformation augmentation."""
    
    def __init__(self, shear_range: float = 0.2) -> None:
        """
        Initialize shear augmentation.
        
        Args:
            shear_range: Maximum shear angle in radians
        """
        self.shear_range = shear_range
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random shear transformation.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Sheared image
        """
        h, w = image.shape[:2]
        shear = np.random.uniform(-self.shear_range, self.shear_range)
        
        pts1 = np.float32([[0, 0], [w, 0], [0, h]])
        pts2 = np.float32([[0, 0], [w, shear*h], [0, h]])
        
        M = cv2.getAffineTransform(pts1, pts2)
        sheared = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return sheared


class RandomBrightness(AugmentationBase):
    """Random brightness adjustment augmentation."""
    
    def __init__(self, brightness_range: float = 0.2) -> None:
        """
        Initialize brightness augmentation.
        
        Args:
            brightness_range: Adjustment range as fraction of max value
        """
        self.brightness_range = brightness_range
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random brightness adjustment.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Brightness-adjusted image
        """
        # Convert to LAB to adjust brightness (L channel)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        adjustment = np.random.uniform(-self.brightness_range, self.brightness_range) * 255
        lab[:, :, 0] += adjustment
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return result


class RandomContrast(AugmentationBase):
    """Random contrast adjustment augmentation."""
    
    def __init__(self, contrast_range: float = 0.2) -> None:
        """
        Initialize contrast augmentation.
        
        Args:
            contrast_range: Adjustment range as fraction
        """
        self.contrast_range = contrast_range
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random contrast adjustment.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Contrast-adjusted image
        """
        factor = np.random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
        mean = np.mean(image, axis=(0, 1))
        
        adjusted = (image.astype(np.float32) - mean) * factor + mean
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted


class RandomNoise(AugmentationBase):
    """Add random Gaussian noise augmentation."""
    
    def __init__(self, noise_std: float = 10.0) -> None:
        """
        Initialize noise augmentation.
        
        Args:
            noise_std: Standard deviation of Gaussian noise
        """
        self.noise_std = noise_std
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Add random Gaussian noise.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Image with added noise
        """
        noise = np.random.normal(0, self.noise_std, image.shape)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy


class DataAugmentor:
    """
    Main data augmentation pipeline for pneumonia detection.
    
    Combines multiple augmentation techniques with configurable probability
    and intensity.
    """
    
    def __init__(
        self,
        rotation_range: int = 20,
        width_shift_range: float = 0.2,
        height_shift_range: float = 0.2,
        shear_range: float = 0.2,
        zoom_range: float = 0.2,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        brightness_range: float = 0.2,
        contrast_range: float = 0.2,
        noise_std: float = 5.0,
        augmentation_probability: float = 0.5
    ) -> None:
        """
        Initialize data augmentor.
        
        Args:
            rotation_range: Max rotation angle in degrees
            width_shift_range: Max horizontal shift as fraction
            height_shift_range: Max vertical shift as fraction
            shear_range: Max shear angle
            zoom_range: Max zoom factor
            horizontal_flip: Enable horizontal flipping
            vertical_flip: Enable vertical flipping
            brightness_range: Brightness adjustment range
            contrast_range: Contrast adjustment range
            noise_std: Standard deviation of noise
            augmentation_probability: Probability of applying augmentation
        """
        self.augmentation_probability = augmentation_probability
        
        self.augmentations: list[AugmentationBase] = [
            aug for aug in [
                RandomRotation(rotation_range),
                RandomShift(width_shift_range) if width_shift_range > 0 else None,
                RandomZoom(zoom_range) if zoom_range > 0 else None,
                RandomFlip(horizontal_flip, vertical_flip),
                RandomShear(shear_range) if shear_range > 0 else None,
                RandomBrightness(brightness_range) if brightness_range > 0 else None,
                RandomContrast(contrast_range) if contrast_range > 0 else None,
                RandomNoise(noise_std) if noise_std > 0 else None,
            ] if aug is not None
        ]
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to image.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Augmented image
        """
        if np.random.rand() > self.augmentation_probability:
            return image
        
        # Randomly select 1-3 augmentations to apply
        num_augmentations = np.random.randint(1, min(4, len(self.augmentations) + 1))
        augmentations = np.random.choice(
            self.augmentations, 
            size=num_augmentations, 
            replace=False
        )
        
        augmented = image.copy()
        for augmentation in augmentations:
            augmented = augmentation.apply(augmented)
        
        return augmented
    
    def batch_augment(
        self, 
        images: np.ndarray,
        augmentation_factor: int = 1
    ) -> np.ndarray:
        """
        Apply augmentations to a batch of images.
        
        Args:
            images: Batch of images (batch_size, height, width, channels)
            augmentation_factor: Number of augmented versions per image
            
        Returns:
            np.ndarray: Batch with augmented images
        """
        augmented_batch = []
        
        for image in images:
            augmented_batch.append(image)  # Keep original
            
            for _ in range(augmentation_factor):
                augmented = self.augment(image)
                augmented_batch.append(augmented)
        
        result = np.array(augmented_batch)
        logger.info(
            f"Augmented batch: {len(images)} images -> "
            f"{len(augmented_batch)} images (factor: {augmentation_factor})"
        )
        return result
