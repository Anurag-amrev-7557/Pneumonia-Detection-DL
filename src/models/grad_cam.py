"""
Grad-CAM visualization for model interpretability.

Provides visualization of which regions in medical images the model
focuses on for making pneumonia detection predictions.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
try:
    import keras
except ImportError:
    from tensorflow import keras

from src.utils.image_processing import ImageProcessor

logger = logging.getLogger(__name__)


class GradCAMVisualizer:
    """
    Generate Grad-CAM (Gradient-weighted Class Activation Map) visualizations.
    
    Helps interpret model predictions by highlighting important regions
    in the input image.
    """
    
    def __init__(
        self,
        model: keras.Model,
        layer_name: str | None = None,
        image_size: tuple[int, int] = (224, 224)
    ) -> None:
        """
        Initialize Grad-CAM visualizer.
        
        Args:
            model: Keras model to visualize
            layer_name: Name of layer to visualize
            image_size: Target image size
        """
        self.model = model
        self.image_processor = ImageProcessor(target_size=image_size)
        self.image_size = image_size
        
        # Check if the model contains a nested backbone (e.g., ResNet50, VGG16)
        self.backbone_layer = None
        self.head_layers = []
        for idx, layer in enumerate(self.model.layers):
            if hasattr(layer, 'layers') and any("conv" in l.name.lower() for l in layer.layers):
                self.backbone_layer = layer
                self.head_layers = self.model.layers[idx + 1:]
                break

        if self.backbone_layer is not None:
            if layer_name is None:
                for layer in reversed(self.backbone_layer.layers):
                    shape = getattr(layer, 'output_shape', None)
                    if shape is None and hasattr(layer, 'output'):
                        shape = getattr(layer.output, 'shape', None)
                    if ("conv" in layer.name.lower() or "out" in layer.name.lower()) and shape is not None and len(shape) == 4:
                        layer_name = layer.name
                        break
            self.layer_name = layer_name
            self.target_layer = self.backbone_layer.get_layer(layer_name)
            backbone_input = self.backbone_layer.input if hasattr(self.backbone_layer, 'input') else self.backbone_layer.inputs[0]
            self.grad_model = keras.models.Model(
                inputs=backbone_input,
                outputs=[self.target_layer.output, self.backbone_layer.output]
            )
        else:
            if layer_name is None:
                layer_name = self._find_last_conv_layer()
            self.layer_name = layer_name
            self.target_layer = self.model.get_layer(layer_name)
            self.grad_model = keras.models.Model(
                inputs=self.model.input,
                outputs=[self.model.output, self.target_layer.output]
            )
        
        logger.info(f"Initialized Grad-CAM for layer: {self.layer_name}")
    
    def _find_last_conv_layer(self) -> str:
        """Find the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            if "conv" in layer.name.lower():
                return layer.name
        
        raise ValueError("No convolutional layer found in model")
    
    def generate_heatmap(
        self,
        image: str | Path | np.ndarray,
        class_index: int = 1,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image: Input image path or array
            class_index: Class index to generate attention for
            normalize: Whether to normalize heatmap to [0, 1]
            
        Returns:
            np.ndarray: Heatmap (height, width)
        """
        # Preprocess image
        if isinstance(image, (str, Path)):
            img_array = self.image_processor.preprocess_image(image)
        else:
            img_array = self._preprocess_array(image)
        
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        
        with tf.GradientTape() as tape:
            if self.backbone_layer is not None:
                conv_outputs, backbone_out = self.grad_model(img_batch)
                tape.watch(conv_outputs)
                x = backbone_out
                for head_layer in self.head_layers:
                    x = head_layer(x)
                predictions = x
            else:
                predictions, conv_outputs = self.grad_model(img_batch)
            class_channel = predictions[:, class_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Average over spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ tf.expand_dims(pooled_grads, -1)
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU
        heatmap = tf.nn.relu(heatmap)
        
        # Normalize
        if normalize:
            max_heat = tf.reduce_max(heatmap)
            if max_heat > 0:
                heatmap = heatmap / max_heat
        
        return heatmap.numpy()
    
    def overlay_heatmap(
        self,
        image: str | Path | np.ndarray,
        heatmap: np.ndarray | None = None,
        class_index: int = 1,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image
            heatmap: Pre-computed heatmap (if None, generates it)
            class_index: Class index for heatmap generation
            alpha: Transparency of overlay (0-1)
            colormap: OpenCV colormap
            
        Returns:
            np.ndarray: Image with overlay (RGB, 0-255 uint8)
        """
        # Load original image
        if isinstance(image, (str, Path)):
            original_img = self.image_processor.load_image(image)
        else:
            original_img = self.image_processor.load_image_from_array(image)
        
        # Resize to match
        original_img = self.image_processor.resize_image(original_img)
        
        # Generate heatmap if needed
        if heatmap is None:
            heatmap = self.generate_heatmap(image, class_index)
        
        # Resize heatmap to match image
        h, w = original_img.shape[:2]
        heatmap_resized = cv2.resize(
            heatmap,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize heatmap to 0-255
        heatmap_scaled = (heatmap_resized * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_scaled, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend images
        overlay = cv2.addWeighted(
            original_img,
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )
        
        return overlay
    
    def visualize_and_save(
        self,
        image_path: str | Path,
        output_path: str | Path,
        class_index: int = 1,
        include_original: bool = True,
        include_heatmap: bool = True
    ) -> None:
        """
        Generate and save visualization.
        
        Args:
            image_path: Input image path
            output_path: Output image path
            class_index: Class index to visualize
            include_original: Whether to include original in output
            include_heatmap: Whether to include pure heatmap
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load original image
        original_img = self.image_processor.load_image(image_path)
        original_img = self.image_processor.resize_image(original_img)
        
        # Generate heatmap
        heatmap = self.generate_heatmap(image_path, class_index)
        
        # Create overlay
        overlay = self.overlay_heatmap(image_path, heatmap, class_index)
        
        # Combine images
        if include_original and include_heatmap:
            h, w = original_img.shape[:2]
            heatmap_3ch = np.stack([heatmap]*3, axis=-1)
            heatmap_3ch = (heatmap_3ch * 255).astype(np.uint8)
            heatmap_resized = cv2.resize(heatmap_3ch, (w, h))
            
            combined = np.hstack([original_img, heatmap_resized, overlay])
        elif include_original:
            combined = np.hstack([original_img, overlay])
        else:
            combined = overlay
        
        # Save
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), combined_bgr)
        
        logger.info(f"Saved visualization to {output_path}")
    
    def _preprocess_array(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image array."""
        image = self.image_processor.load_image_from_array(image)
        image = self.image_processor.resize_image(image)
        image = self.image_processor.normalize_image(image)
        return image


class SaliencyMapVisualizer:
    """
    Generate saliency maps showing pixel-level importance.
    """
    
    def __init__(
        self,
        model: keras.Model,
        image_size: tuple[int, int] = (224, 224)
    ) -> None:
        """
        Initialize saliency visualizer.
        
        Args:
            model: Keras model
            image_size: Target image size
        """
        self.model = model
        self.image_processor = ImageProcessor(target_size=image_size)
    
    def generate_saliency_map(
        self,
        image: str | Path | np.ndarray,
        class_index: int = 1
    ) -> np.ndarray:
        """
        Generate saliency map showing gradient magnitude at each pixel.
        
        Args:
            image: Input image
            class_index: Target class index
            
        Returns:
            np.ndarray: Saliency map
        """
        # Preprocess image
        if isinstance(image, (str, Path)):
            img_array = self.image_processor.preprocess_image(image)
        else:
            img_array = self._preprocess_array(image)
        
        # Convert to tensor
        img_tensor = tf.convert_to_tensor(
            np.expand_dims(img_array, axis=0),
            dtype=tf.float32
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = self.model(img_tensor)
            target = predictions[:, class_index]
        
        grads = tape.gradient(target, img_tensor)
        
        # Compute saliency (magnitude of gradients)
        saliency = tf.reduce_max(tf.abs(grads), axis=-1)
        saliency = saliency[0].numpy()
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def visualize_saliency(
        self,
        image: str | Path | np.ndarray,
        class_index: int = 1,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay saliency map on image.
        
        Args:
            image: Input image
            class_index: Target class index
            alpha: Transparency
            
        Returns:
            np.ndarray: Image with saliency overlay
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = self.image_processor.load_image(image)
        else:
            img = self.image_processor.load_image_from_array(image)
        
        img = self.image_processor.resize_image(img)
        
        # Generate saliency map
        saliency = self.generate_saliency_map(image, class_index)
        
        # Resize to match image
        h, w = img.shape[:2]
        saliency_resized = cv2.resize(
            saliency,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Convert to heatmap
        saliency_heatmap = (saliency_resized * 255).astype(np.uint8)
        saliency_colored = cv2.applyColorMap(saliency_heatmap, cv2.COLORMAP_HOT)
        saliency_colored = cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB)
        
        # Blend
        overlay = cv2.addWeighted(img, 1-alpha, saliency_colored, alpha, 0)
        
        return overlay
    
    def _preprocess_array(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image array."""
        image = self.image_processor.load_image_from_array(image)
        image = self.image_processor.resize_image(image)
        image = self.image_processor.normalize_image(image)
        return image
