"""
Configuration management for the pneumonia detection application.

This module provides centralized configuration handling for model training,
inference, and data processing with environment-based overrides.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data handling and preprocessing."""
    
    data_dir: Path = Path("data")
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    image_size: tuple = (224, 224)
    batch_size: int = 32
    random_seed: int = 42
    augmentation_enabled: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not (self.train_split + self.val_split + self.test_split) == 1.0:
            raise ValueError("Train, validation, and test splits must sum to 1.0")
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("Image size dimensions must be positive")


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    
    model_type: str = "resnet50"  # default production architecture
    input_shape: tuple = (224, 224, 3)
    num_classes: int = 2
    dropout_rate: float = 0.5
    learning_rate: float = 0.0001
    epochs: int = 25
    early_stopping_patience: int = 6
    validation_split: float = 0.2
    optimizer: str = "adam"  # adam, sgd, rmsprop
    loss_function: str = "binary_crossentropy"
    metrics: list | None = None
    
    def __post_init__(self) -> None:
        """Initialize default metrics if not provided."""
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "auc"]
        if self.dropout_rate < 0 or self.dropout_rate >= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    
    model_path: Path = Path("models/current/best_model.h5")
    secondary_model_path: Path = Path("models/current/densenet121_best.h5")
    confidence_threshold: float = 0.5
    enable_grad_cam: bool = True
    grad_cam_layer: str | None = None
    batch_inference: bool = True
    
    def __post_init__(self) -> None:
        """Validate inference configuration."""
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1")


@dataclass
class UIConfig:
    """Configuration for the web interface."""
    
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_size: int = 10 * 1024 * 1024  # 10 MB
    upload_folder: Path = Path("uploads")
    allowed_extensions: set | None = None
    
    def __post_init__(self) -> None:
        """Initialize default allowed extensions."""
        if self.allowed_extensions is None:
            self.allowed_extensions = {"png", "jpg", "jpeg", "gif"}


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    
    rotation_range: int = 20
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    shear_range: float = 0.2
    zoom_range: float = 0.2
    horizontal_flip: bool = True
    vertical_flip: bool = False
    fill_mode: str = "nearest"


class Settings:
    """
    Main settings class that aggregates all configuration sections.
    
    Supports environment variable overrides via:
    - APP_ENV: environment (development, production, testing)
    - APP_DEBUG: debug mode (true/false)
    - APP_LOG_LEVEL: logging level (DEBUG, INFO, WARNING, ERROR)
    """
    
    def __init__(self) -> None:
        """Initialize all configuration sections."""
        self.data = DataConfig()
        self.model = ModelConfig()
        self.inference = InferenceConfig()
        self.ui = UIConfig()
        self.augmentation = AugmentationConfig()
        
        self.env = os.getenv("APP_ENV", "development")
        self.debug = os.getenv("APP_DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("APP_LOG_LEVEL", "INFO")
        
        self._load_from_env()
        self._validate()
        
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Data config overrides
        if data_dir := os.getenv("DATA_DIR"):
            self.data.data_dir = Path(data_dir)
        if batch_size := os.getenv("BATCH_SIZE"):
            self.data.batch_size = int(batch_size)
            
        # Model config overrides
        if model_type := os.getenv("MODEL_TYPE"):
            self.model.model_type = model_type
        if learning_rate := os.getenv("LEARNING_RATE"):
            self.model.learning_rate = float(learning_rate)
        if epochs := os.getenv("EPOCHS"):
            self.model.epochs = int(epochs)
            
        # Inference config overrides
        if model_path := os.getenv("MODEL_PATH"):
            self.inference.model_path = Path(model_path)
        if threshold := os.getenv("CONFIDENCE_THRESHOLD"):
            self.inference.confidence_threshold = float(threshold)
            
        # UI config overrides
        if port := os.getenv("UI_PORT"):
            self.ui.port = int(port)
        if upload_size := os.getenv("MAX_UPLOAD_SIZE"):
            self.ui.max_upload_size = int(upload_size)
            
    def _validate(self) -> None:
        """Validate all configuration sections."""
        try:
            # Validation happens in dataclass __post_init__ methods
            logger.info(f"Configuration loaded successfully (env={self.env})")
        except ValueError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
    
    def get_config_dict(self) -> dict:
        """
        Get all configuration as a dictionary.
        
        Returns:
            dict: Complete configuration dictionary
        """
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "inference": self.inference.__dict__,
            "ui": self.ui.__dict__,
            "augmentation": self.augmentation.__dict__,
            "environment": self.env,
            "debug": self.debug,
            "log_level": self.log_level,
        }


# Global settings instance
settings = Settings()
