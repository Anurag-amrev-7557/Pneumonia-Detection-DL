"""
Deep learning model architectures for pneumonia detection.

This module provides several pre-trained and custom CNN architectures
optimized for medical image classification.
"""

import logging
from typing import ClassVar

from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.applications import InceptionV3, ResNet50, VGG16

logger = logging.getLogger(__name__)


class CustomCNNArchitecture:
    """
    Custom lightweight CNN architecture for pneumonia detection.
    
    Optimized for medical imaging with reasonable computational efficiency
    and high accuracy.
    """
    
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 2,
        dropout_rate: float = 0.5
    ) -> None:
        """
        Initialize custom CNN architecture.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        if len(input_shape) != 3 or any(d <= 0 for d in input_shape):
            raise ValueError("Input shape must have 3 positive dimensions")

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    def build(self) -> Model:
        """
        Build the custom CNN model.
        
        Returns:
            keras.Model: Compiled model
        """
        model = Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=self.input_shape, name='conv2d_1'),
            layers.BatchNormalization(name='batch_norm_1'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         name='conv2d_2'),
            layers.BatchNormalization(name='batch_norm_2'),
            layers.MaxPooling2D((2, 2), name='maxpool_1'),
            layers.Dropout(self.dropout_rate, name='dropout_1'),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                         name='conv2d_3'),
            layers.BatchNormalization(name='batch_norm_3'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                         name='conv2d_4'),
            layers.BatchNormalization(name='batch_norm_4'),
            layers.MaxPooling2D((2, 2), name='maxpool_2'),
            layers.Dropout(self.dropout_rate, name='dropout_2'),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                         name='conv2d_5'),
            layers.BatchNormalization(name='batch_norm_5'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                         name='conv2d_6'),
            layers.BatchNormalization(name='batch_norm_6'),
            layers.MaxPooling2D((2, 2), name='maxpool_3'),
            layers.Dropout(self.dropout_rate, name='dropout_3'),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', 
                         name='conv2d_7'),
            layers.BatchNormalization(name='batch_norm_7'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', 
                         name='conv2d_8'),
            layers.BatchNormalization(name='batch_norm_8'),
            layers.MaxPooling2D((2, 2), name='maxpool_4'),
            layers.Dropout(self.dropout_rate, name='dropout_4'),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(name='global_avg_pool'),
            
            # Fully Connected Layers
            layers.Dense(512, activation='relu', name='dense_1'),
            layers.BatchNormalization(name='batch_norm_9'),
            layers.Dropout(self.dropout_rate, name='dropout_5'),
            
            layers.Dense(256, activation='relu', name='dense_2'),
            layers.BatchNormalization(name='batch_norm_10'),
            layers.Dropout(self.dropout_rate, name='dropout_6'),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax', name='output'),
        ], name='custom_cnn')
        
        logger.info("Built custom CNN architecture")
        return model


class TransferLearningArchitecture:
    """Base class for transfer learning models."""
    
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        freeze_base: bool = True
    ) -> None:
        """
        Initialize transfer learning architecture.
        
        Args:
            input_shape: Shape of input images
            num_classes: Number of output classes
            dropout_rate: Dropout rate for custom layers
            freeze_base: Whether to freeze base model weights
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.freeze_base = freeze_base
    
    def _build_custom_top(self, base_model: Model) -> Model:
        """
        Build custom top layers on base model.
        
        Args:
            base_model: Pre-trained base model
            
        Returns:
            keras.Model: Complete model with custom top
        """
        # Freeze base model if requested
        base_model.trainable = not self.freeze_base
        
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        
        # Custom top - stable, regularized classification head
        x = layers.GlobalAveragePooling2D(name='custom_gap')(x)
        x = layers.Dense(256, activation='relu', name='custom_dense_1')(x)
        top_dropout = min(self.dropout_rate, 0.3)
        x = layers.Dropout(top_dropout, name='custom_dropout_1')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', 
                              name='output')(x)
        
        model = Model(inputs, outputs)
        return model


class VGG16Architecture(TransferLearningArchitecture):
    """VGG16-based architecture for pneumonia detection."""
    
    def build(self) -> Model:
        """
        Build VGG16 model.
        
        Returns:
            keras.Model: VGG16 model with custom top
        """
        # Load pre-trained VGG16
        base_model = VGG16(
            weights='imagenet',
            input_shape=self.input_shape,
            include_top=False
        )
        
        model = self._build_custom_top(base_model)
        logger.info("Built VGG16 transfer learning architecture")
        return model


class ResNet50Architecture(TransferLearningArchitecture):
    """ResNet50-based architecture for pneumonia detection."""
    
    def build(self) -> Model:
        """
        Build ResNet50 model.
        
        Returns:
            keras.Model: ResNet50 model with custom top
        """
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            input_shape=self.input_shape,
            include_top=False
        )
        
        model = self._build_custom_top(base_model)
        logger.info("Built ResNet50 transfer learning architecture")
        return model


class InceptionV3Architecture(TransferLearningArchitecture):
    """InceptionV3-based architecture for pneumonia detection."""
    
    def build(self) -> Model:
        """
        Build InceptionV3 model.
        
        Returns:
            keras.Model: InceptionV3 model with custom top
        """
        # Load pre-trained InceptionV3
        base_model = InceptionV3(
            weights='imagenet',
            input_shape=self.input_shape,
            include_top=False
        )
        
        model = self._build_custom_top(base_model)
        logger.info("Built InceptionV3 transfer learning architecture")
        return model


class ModelFactory:
    """Factory for creating model architectures."""
    
    _architectures: ClassVar = {
        'custom_cnn': CustomCNNArchitecture,
        'vgg16': VGG16Architecture,
        'resnet50': ResNet50Architecture,
        'inceptionv3': InceptionV3Architecture,
    }
    
    @classmethod
    def create(
        cls,
        model_type: str,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        **kwargs
    ) -> Model:
        """
        Create a model architecture.
        
        Args:
            model_type: Type of model ('custom_cnn', 'vgg16', 'resnet50', 'inceptionv3')
            input_shape: Shape of input images
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            **kwargs: Additional arguments for specific architectures
            
        Returns:
            keras.Model: Built model
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls._architectures:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported types: {list(cls._architectures.keys())}"
            )
        
        architecture_class = cls._architectures[model_type]
        architecture = architecture_class(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            **kwargs
        )
        
        model = architecture.build()
        logger.info(f"Created {model_type} model")
        return model
    
    @classmethod
    def get_supported_models(cls) -> list:
        """
        Get list of supported model types.
        
        Returns:
            list: Supported model type names
        """
        return list(cls._architectures.keys())


def unfreeze_layers(model: Model, num_layers: int = -1) -> None:
    """
    Unfreeze layers in a model for fine-tuning.
    
    Args:
        model: Keras model
        num_layers: Number of layers to unfreeze from the end (-1 for all)
    """
    if num_layers == -1:
        num_layers = len(model.layers)
    
    for layer in model.layers[-num_layers:]:
        layer.trainable = True
    
    logger.info(f"Unfroze last {num_layers} layers for fine-tuning")


def get_model_summary(model: Model) -> str:
    """
    Get a text summary of model architecture.
    
    Args:
        model: Keras model
        
    Returns:
        str: Model summary as string
    """
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        model.summary()
    return f.getvalue()
