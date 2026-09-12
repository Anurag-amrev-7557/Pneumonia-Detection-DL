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
MODELS_DIR = Path("models/current")

# Map of local filename → filename on HF Hub
MODEL_FILES = {
    "best_model.h5": "best_model.h5",
    "densenet121_best.h5": "densenet121_best.h5",
}


def ensure_models_downloaded() -> bool:
    """
    Download model weights from HF Hub if they are not present locally.

    Returns:
        bool: True if all required models are available after the call.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_present = all((MODELS_DIR / fname).exists() for fname in MODEL_FILES)
    if all_present:
        logger.info("Model weights already present — skipping download.")
        return True

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

    return success
