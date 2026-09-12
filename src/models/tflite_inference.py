"""
Lightweight TFLite-based inference engine for Streamlit Cloud deployment.

Uses TensorFlow's built-in tf.lite module for model inference,
with minimal memory footprint compared to full model loading.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image
import tensorflow as tf

logger = logging.getLogger(__name__)


class TFLiteDetector:
    """Pneumonia detector using TFLite models (RAM-efficient for Streamlit Cloud)."""

    CLASSES = {0: "Normal", 1: "Pneumonia"}

    def __init__(
        self,
        model_path: str | Path,
        secondary_model_path: str | Path | None = None,
        image_size: tuple[int, int] = (224, 224),
        confidence_threshold: float = 0.5,
    ) -> None:
        """
        Initialize TFLite detector.

        Args:
            model_path: Path to primary .tflite model file
            secondary_model_path: Path to secondary .tflite model (optional, for ensemble)
            image_size: Target image size for input
            confidence_threshold: Decision threshold for Pneumonia class
        """
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold

        # Load primary model
        self.interpreter = self._load_tflite_model(self.model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Optional secondary model for ensemble
        self.secondary_interpreter = None
        if secondary_model_path and Path(secondary_model_path).exists():
            try:
                self.secondary_interpreter = self._load_tflite_model(secondary_model_path)
                logger.info(f"Loaded secondary TFLite model for ensemble: {secondary_model_path}")
            except Exception as e:
                logger.warning(f"Could not load secondary model: {e}")

        logger.info(f"Initialized TFLite detector with model: {self.model_path}")

    def _load_tflite_model(self, model_path: Path):
        """Load a TFLite model file using TensorFlow's lite interpreter."""
        try:
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            logger.info(f"Loaded TFLite model: {model_path}")
            return interpreter
        except Exception as e:
            logger.error(f"Failed to load TFLite model {model_path}: {e}")
            raise

    def predict(
        self,
        image_input: str | Path | Any,
        return_probabilities: bool = True,
        return_gradcam: bool = False,   # accepted but ignored — TFLite has no GradCAM
        confidence_threshold: float | None = None,
        crop_margins: bool = False,
    ) -> Dict[str, Any]:
        """
        Make prediction on image input.

        Args:
            image_input: Path to image file or PIL Image
            return_probabilities: Whether to return class probabilities
            return_gradcam: Accepted for API compatibility; ignored (TFLite has no GradCAM)
            confidence_threshold: Optional override for decision threshold
            crop_margins: If True, crops 6%/7% from edges to remove lead markers

        Returns:
            dict: Prediction result with label, confidence, probabilities
        """
        start_time = time.perf_counter()

        # Load image
        try:
            if isinstance(image_input, (str, Path)):
                pil_img = Image.open(str(image_input)).convert("RGB")
            elif isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGB")
            else:
                pil_img = Image.fromarray(image_input).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise ValueError(f"Could not process image: {e}")

        # Optional margin crop to remove radiograph lead markers
        if crop_margins:
            w, h = pil_img.size
            left   = int(w * 0.07)
            right  = int(w * 0.93)
            top    = int(h * 0.06)
            bottom = int(h * 0.94)
            pil_img = pil_img.crop((left, top, right, bottom))

        # Resize and normalize
        img_resized = pil_img.resize(self.image_size, Image.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0

        # Add batch dimension
        input_data = np.expand_dims(img_array, axis=0)

        # Primary model prediction
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        primary_output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        # Ensemble with secondary model if available
        if self.secondary_interpreter is not None:
            self.secondary_interpreter.set_tensor(
                self.secondary_interpreter.get_input_details()[0]["index"],
                input_data,
            )
            self.secondary_interpreter.invoke()
            secondary_output = self.secondary_interpreter.get_tensor(
                self.secondary_interpreter.get_output_details()[0]["index"]
            )[0]
            # Equal-weight soft voting
            final_probs = 0.5 * primary_output + 0.5 * secondary_output
            model_name = "ResNet-50 + DenseNet-121 (TFLite Ensemble)"
        else:
            final_probs = primary_output
            model_name = "ResNet-50 (TFLite)"

        # Format output
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
        active_threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        # Probabilities are already normalized (softmax output from TFLite)
        p_pneumonia = float(final_probs[1])
        is_pneumonia = bool(p_pneumonia >= active_threshold)
        predicted_class = 1 if is_pneumonia else 0
        confidence = p_pneumonia if is_pneumonia else float(final_probs[0])

        result = {
            "label": self.CLASSES.get(predicted_class, "Unknown"),
            "class": self.CLASSES.get(predicted_class, "Unknown"),
            "class_index": predicted_class,
            "confidence": confidence,
            "is_pneumonia": is_pneumonia,
            "is_confident": bool(confidence >= 0.65),
            "is_borderline": bool(0.35 <= p_pneumonia <= 0.65),
            "threshold_used": active_threshold,
            "model_name": model_name,
            "processing_time": elapsed_ms,
            "crop_margins_applied": crop_margins,
            # GradCAM not supported in TFLite — return None so UI degrades gracefully
            "gradcam": None,
            "raw_heatmap": None,
        }

        if return_probabilities:
            result["probabilities"] = {
                self.CLASSES[i]: float(final_probs[i]) for i in range(len(final_probs))
            }

        return result
