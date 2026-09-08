"""
Model inference engine for pneumonia detection predictions.

Handles model loading, batch prediction, confidence calculation, and
result formatting.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import logging
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import tensorflow as tf
try:
    import keras
except ImportError:
    from tensorflow import keras

from src.utils.image_processing import ImageProcessor

logger = logging.getLogger(__name__)


class PneumoniaDetector:
    """
    Inference engine for pneumonia detection.
    
    Loads pre-trained models and provides methods for making predictions
    on single images or batches.
    """
    
    # Class indices mapping
    CLASSES: ClassVar[dict[int, str]] = {0: "Normal", 1: "Pneumonia"}
    
    def __init__(
        self,
        model_path: str | Path,
        image_size: tuple[int, int] = (224, 224),
        confidence_threshold: float = 0.5
    ) -> None:
        """
        Initialize detector.
        
        Args:
            model_path: Path to trained model file
            image_size: Target image size
            confidence_threshold: Confidence threshold for predictions
            
        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model loading fails
        """
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.image_processor = ImageProcessor(target_size=image_size)
        
        self.model = self._load_model()
        
        # Check for secondary ensemble model (CheXNet Dual-Backbone)
        self.secondary_model = None
        densenet_path = self.model_path.parent / "densenet121_best.h5"
        if densenet_path.exists() and densenet_path != self.model_path:
            try:
                self.secondary_model = keras.models.load_model(str(densenet_path), compile=False)
                logger.info(f"Loaded secondary ensemble model for CheXNet dual inference: {densenet_path}")
            except Exception as e:
                logger.warning(f"Could not load secondary ensemble model: {e}")
                
        self.gradcam_visualizer = None
        logger.info(f"Initialized detector with model: {self.model_path}")
    
    def _load_model(self) -> 'keras.Model':
        """
        Load model from file.
        
        Returns:
            Loaded model
            
        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            model = keras.models.load_model(str(self.model_path), compile=False)
            logger.info(f"Loaded model from {self.model_path}")
            return model
        except (OSError, ValueError, RuntimeError) as e:
            raise ValueError(f"Failed to load model: {e}")

    def predict(
        self,
        image_input: str | Path | Any,
        return_probabilities: bool = True,
        return_gradcam: bool = False,
        confidence_threshold: float | None = None,
        crop_margins: bool = False
    ) -> dict[str, Any]:
        """
        Unified predict method compatible with Streamlit file uploaders, 
        file paths, and numpy arrays. Supports Grad-CAM attention, clinical
        sensitivity thresholding, and margin artifact cropping.
        """
        start_time = time.perf_counter()
        from PIL import Image
        
        # Resolve raw RGB image array
        original_rgb = None
        try:
            if isinstance(image_input, (str, Path)):
                img_path = Path(image_input)
                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")
                pil_img = Image.open(str(img_path)).convert("RGB")
                original_rgb = np.array(pil_img)
            elif hasattr(image_input, "read"):
                if hasattr(image_input, "seek"):
                    image_input.seek(0)
                pil_img = Image.open(image_input).convert("RGB")
                original_rgb = np.array(pil_img)
            elif isinstance(image_input, Image.Image):
                original_rgb = np.array(image_input.convert("RGB"))
            elif isinstance(image_input, np.ndarray):
                original_rgb = self.image_processor.load_image_from_array(image_input)
            else:
                pil_img = Image.open(str(image_input)).convert("RGB")
                original_rgb = np.array(pil_img)
        except Exception as e:
            logger.error(f"Error reading image input in predict(): {e}")
            raise ValueError(f"Could not process image input: {e}")

        # Optional margin cropping to remove lead markers ("R", "L") & peripheral stamp artifacts
        if crop_margins:
            h, w = original_rgb.shape[:2]
            h_crop = max(1, int(h * 0.06))
            w_crop = max(1, int(w * 0.07))
            original_rgb = original_rgb[h_crop:h - h_crop, w_crop:w - w_crop]

        # Preprocess array (224x224, normalized to [0, 1])
        preprocessed = self.image_processor.resize_image(original_rgb)
        preprocessed = self.image_processor.normalize_image(preprocessed, method="01")
        batch = np.expand_dims(preprocessed, axis=0)

        # Primary backbone prediction (ResNet-50)
        p_res = self.model(batch, training=False).numpy()
        final_probs = p_res
        
        breakdown = {
            "ResNet-50": {
                self.CLASSES[i]: float(p_res[0][i]) for i in range(len(self.CLASSES))
            }
        }

        # Secondary backbone prediction (DenseNet-121)
        if self.secondary_model is not None:
            p_dense = self.secondary_model(batch, training=False).numpy()
            final_probs = 0.5 * p_res + 0.5 * p_dense
            breakdown["DenseNet-121"] = {
                self.CLASSES[i]: float(p_dense[0][i]) for i in range(len(self.CLASSES))
            }

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
        active_threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        result = self._format_prediction(final_probs[0], return_probabilities, threshold=active_threshold)

        result["model_name"] = "ResNet-50 + DenseNet-121 Ensemble (CheXNet Dual-Backbone)" if self.secondary_model is not None else "ResNet-50"
        result["breakdown"] = breakdown
        result["processing_time"] = elapsed_ms
        result["crop_margins_applied"] = crop_margins
        
        # Check if prediction is in the borderline/indeterminate zone (within +/- 15% of 0.50)
        p_pneu = result["probabilities"]["Pneumonia"] if "probabilities" in result else (1.0 if result["is_pneumonia"] else 0.0)
        result["is_borderline"] = bool(0.35 <= p_pneu <= 0.65)

        # Generate Grad-CAM attention map if requested
        if return_gradcam:
            try:
                if self.gradcam_visualizer is None:
                    from src.models.grad_cam import GradCAMVisualizer
                    self.gradcam_visualizer = GradCAMVisualizer(self.model)
                heatmap = self.gradcam_visualizer.generate_heatmap(original_rgb, class_index=1)
                overlay = self.gradcam_visualizer.overlay_heatmap(original_rgb, heatmap=heatmap, class_index=1, alpha=0.45)
                result["gradcam"] = overlay
                result["raw_heatmap"] = heatmap
                result["input_image_rgb"] = original_rgb
            except Exception as e:
                logger.warning(f"Grad-CAM generation failed: {e}")

        return result

    def predict_image(
        self,
        image_path: str | Path,
        return_probabilities: bool = True
    ) -> dict[str, Any]:
        """Predict on a single image using 0-1 scaling to match training."""
        return self.predict(image_path, return_probabilities=return_probabilities)
    
    def predict_array(
        self,
        image_array: np.ndarray,
        return_probabilities: bool = True
    ) -> dict[str, Any]:
        """Predict on image from numpy array using 0-1 scaling."""
        return self.predict(image_array, return_probabilities=return_probabilities)
    
    def predict_batch(
        self,
        image_paths: list[str | Path],
        return_probabilities: bool = True,
        confidence_threshold: float | None = None,
        crop_margins: bool = False
    ) -> list[dict[str, Any]]:
        """Predict on multiple images using 0-1 scaling."""
        results = [
            self.predict(
                p, 
                return_probabilities=return_probabilities, 
                confidence_threshold=confidence_threshold,
                crop_margins=crop_margins
            ) 
            for p in image_paths
        ]
        logger.info(f"Made batch predictions for {len(image_paths)} images")
        return results

    def _format_prediction(
        self,
        prediction: np.ndarray,
        return_probabilities: bool = True,
        threshold: float | None = None
    ) -> dict[str, Any]:
        """
        Format raw model prediction into readable result.
        
        Args:
            prediction: Model output (logits or probabilities)
            return_probabilities: Whether to include probabilities
            threshold: Decision cutoff for Pneumonia classification
            
        Returns:
            dict: Formatted prediction
        """
        decision_threshold = threshold if threshold is not None else self.confidence_threshold
        if prediction.max() > 1.0 or prediction.min() < 0.0:
            probs = tf.nn.softmax(prediction).numpy()
        else:
            probs = prediction
        
        p_pneumonia = float(probs[1])
        is_pneumonia = bool(p_pneumonia >= decision_threshold)
        predicted_class = 1 if is_pneumonia else 0
        confidence = p_pneumonia if is_pneumonia else float(probs[0])
        
        result = {
            "class": self.CLASSES.get(predicted_class, "Unknown"),
            "class_index": predicted_class,
            "confidence": confidence,
            "is_pneumonia": is_pneumonia,
            "is_confident": bool(confidence >= 0.65),
            "threshold_used": decision_threshold,
        }
        
        if return_probabilities:
            result["probabilities"] = {
                self.CLASSES[i]: float(probs[i])
                for i in range(len(probs))
            }
        
        return result
        
        if return_probabilities:
            result["probabilities"] = {
                self.CLASSES[i]: float(probs[i])
                for i in range(len(probs))
            }
        
        return result
    
    def predict_with_uncertainty(
        self,
        image_path: str | Path,
        num_predictions: int = 10
    ) -> dict[str, Any]:
        """Make predictions with uncertainty estimation via Monte Carlo dropout."""
        image = self.image_processor.preprocess_image(
            image_path, 
            normalize=True, 
            normalization_method="01"
        )
        batch = np.expand_dims(image, axis=0)
        
        predictions = []
        for _ in range(num_predictions):
            pred = self.model(batch, training=True)
            predictions.append(pred.numpy()[0])
        
        predictions = np.array(predictions)
        
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        result = self._format_prediction(mean_pred)
        result["uncertainty"] = float(std_pred.max())
        result["std_probabilities"] = {
            self.CLASSES[i]: float(std_pred[i])
            for i in range(len(std_pred))
        }
        
        return result
    
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": str(self.model_path),
            "model_name": self.model.name,
            "input_shape": tuple(self.model.input_shape),
            "output_shape": tuple(self.model.output_shape),
            "total_params": self.model.count_params(),
            "trainable_params": sum(
                w.shape.num_elements() if w.shape.num_elements() else 0
                for w in self.model.trainable_weights
            ),
            "classes": self.CLASSES,
            "confidence_threshold": self.confidence_threshold,
        }


class BatchPredictor:
    """
    Efficient batch prediction manager.
    
    Handles predictions for large batches with progress tracking
    and result aggregation.
    """
    
    def __init__(self, detector: PneumoniaDetector, batch_size: int = 32) -> None:
        """
        Initialize batch predictor.
        
        Args:
            detector: PneumoniaDetector instance
            batch_size: Batch size for predictions
        """
        self.detector = detector
        self.batch_size = batch_size
        self.results = []
    
    def predict_directory(
        self,
        directory: str | Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    ) -> list[dict[str, Any]]:
        """Predict on all images in directory."""
        directory = Path(directory)
        
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory.glob(f"*{ext}"))
        
        if not image_paths:
            logger.warning(f"No images found in {directory}")
            return []
        
        logger.info(f"Found {len(image_paths)} images")
        
        results = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_results = self.detector.predict_batch(batch_paths)
            results.extend(batch_results)
        
        self.results = results
        logger.info(f"Completed predictions for {len(results)} images")
        
        return results
    
    def get_statistics(self) -> dict[str, Any]:
        """Get statistics from batch predictions."""
        if not self.results:
            return {}
        
        pneumonia_count = sum(1 for r in self.results if r["is_pneumonia"])
        normal_count = len(self.results) - pneumonia_count
        avg_confidence = np.mean([r["confidence"] for r in self.results])
        
        return {
            "total_predictions": len(self.results),
            "pneumonia_cases": pneumonia_count,
            "normal_cases": normal_count,
            "pneumonia_percentage": pneumonia_count / len(self.results) * 100,
            "average_confidence": float(avg_confidence),
        }