"""
Model weight loader for cloud deployment.

On Streamlit Community Cloud (and any environment where local .h5 files are
absent) this module downloads weights from Hugging Face Hub into
models/current/ before the app tries to load them.

Local development: if the files already exist they are never re-downloaded.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HF_REPO_ID = "Anurag234/pulmo-ai-weights"

# Use absolute path anchored to this file's location so it works regardless
# of the working directory (important on Streamlit Cloud)
_SRC_UTILS_DIR = Path(__file__).resolve().parent        # src/utils/
_PROJECT_ROOT   = _SRC_UTILS_DIR.parent.parent          # project root
MODELS_DIR      = _PROJECT_ROOT / "models" / "current"

# Bump this when new model files are uploaded to HF Hub.
# Forces re-download even if local .tflite files already exist.
MODEL_VERSION = "v3"

# Map of local filename → filename on HF Hub
# Using TFLite (lightweight) instead of full .h5 for Streamlit Cloud (1GB RAM)
MODEL_FILES = {
    "best_model.tflite": "best_model.tflite",
    "densenet121_best.tflite": "densenet121_best.tflite",
}


def ensure_models_downloaded() -> bool:
    """
    Download model weights from HF Hub if not present or outdated.

    Returns:
        bool: True if all required models are available after the call.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Version sentinel — forces re-download when MODEL_VERSION is bumped
    version_file = MODELS_DIR / ".model_version"
    current_version = version_file.read_text().strip() if version_file.exists() else ""

    all_present = all((MODELS_DIR / fname).exists() for fname in MODEL_FILES)
    if all_present and current_version == MODEL_VERSION:
        logger.info("Model weights already present and up to date — skipping download.")
        return True

    if all_present and current_version != MODEL_VERSION:
        logger.info(f"Model version mismatch ({current_version} vs {MODEL_VERSION}) — re-downloading.")
        for fname in MODEL_FILES:
            (MODELS_DIR / fname).unlink(missing_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub is not installed. "
            "Add it to requirements.txt: huggingface_hub>=0.23.0"
        )
        return False

    success = True
    for local_name, hub_name in MODEL_FILES.items():
        dest = MODELS_DIR / local_name
        if dest.exists():
            logger.info(f"  {local_name} — already present, skipping.")
            continue
        try:
            logger.info(f"  Downloading {hub_name} from {HF_REPO_ID} ...")
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hub_name,
                repo_type="model",
                local_dir=str(MODELS_DIR),
            )
            logger.info(f"  Saved to {downloaded}")
        except Exception as exc:
            logger.error(f"  Failed to download {hub_name}: {exc}")
            success = False

    if success:
        version_file.write_text(MODEL_VERSION)
        logger.info(f"  Wrote version sentinel {MODEL_VERSION}")

    return success
