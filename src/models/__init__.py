"""Model architectures and training utilities."""

# Imports are intentionally deferred — importing these at package level
# pulls in TensorFlow eagerly, which breaks Streamlit Cloud deployment.
# Import specific modules directly where needed:
#   from src.models.tflite_inference import TFLiteDetector  (cloud/lightweight)
#   from src.models.inference import PneumoniaDetector      (local/full TF)
#   from src.models.architectures import ModelFactory       (training only)
