"""
Model weight and sample image loader for cloud deployment.

Downloads TFLite model weights and curated demo images from HF Hub
into the local filesystem on first run.

Local development: if the files already exist they are never re-downloaded.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HF_REPO_ID = "Anurag234/pulmo-ai-weights"

# Absolute paths anchored to this file — works regardless of cwd
_SRC_UTILS_DIR = Path(__file__).resolve().parent        # src/utils/
_PROJECT_ROOT  = _SRC_UTILS_DIR.parent.parent           # project root
MODELS_DIR     = _PROJECT_ROOT / "models" / "current"
SAMPLES_DIR    = _PROJECT_ROOT / "data" / "test"

# Bump this to force a full re-download on next app start
MODEL_VERSION = "v3"

# Model weights: local path relative to MODELS_DIR → HF Hub filename
MODEL_FILES = {
    "best_model.tflite":       "best_model.tflite",
    "densenet121_best.tflite": "densenet121_best.tflite",
}

# Demo sample images: local path relative to project root → HF Hub filename
SAMPLE_FILES = {
    "data/test/NORMAL/IM-0341-0001.jpeg":                    "samples/NORMAL/IM-0341-0001.jpeg",
    "data/test/PNEUMONIA/person1619_bacteria_4261.jpeg":     "samples/PNEUMONIA/person1619_bacteria_4261.jpeg",
    "data/test/PNEUMONIA/person478_virus_975.jpeg":          "samples/PNEUMONIA/person478_virus_975.jpeg",
    "data/test/PNEUMONIA/person1014_bacteria_2945.jpeg":     "samples/PNEUMONIA/person1014_bacteria_2945.jpeg",
}


def _download_file(hf_hub_download, hub_name: str, local_dest: Path) -> bool:
    """Download a single file from HF Hub to local_dest. Returns True on success."""
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"  Downloading {hub_name} ...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hub_name,
            repo_type="model",
            local_dir=str(local_dest.parent),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download saves to local_dir/<filename>, rename if needed
        downloaded = local_dest.parent / Path(hub_name).name
        if downloaded.exists() and downloaded != local_dest:
            downloaded.rename(local_dest)
        logger.info(f"  Saved to {local_dest}")
        return True
    except Exception as exc:
        logger.error(f"  Failed to download {hub_name}: {exc}")
        return False


def ensure_models_downloaded() -> bool:
    """
    Download TFLite model weights from HF Hub if not present or outdated.
    Also downloads curated demo sample images.

    Returns:
        bool: True if all required models are available after the call.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Version sentinel — forces re-download when MODEL_VERSION is bumped
    version_file = MODELS_DIR / ".model_version"
    current_version = version_file.read_text().strip() if version_file.exists() else ""

    models_present = all((MODELS_DIR / fname).exists() for fname in MODEL_FILES)

    if models_present and current_version == MODEL_VERSION:
        logger.info("Model weights up to date — skipping model download.")
        # Still check samples (they don't have a version gate)
        _ensure_samples_downloaded()
        return True

    if models_present and current_version != MODEL_VERSION:
        logger.info(f"Version mismatch ({current_version!r} → {MODEL_VERSION!r}) — re-downloading models.")
        for fname in MODEL_FILES:
            (MODELS_DIR / fname).unlink(missing_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed. Add huggingface_hub>=0.23.0 to requirements.txt")
        return False

    success = True
    for local_name, hub_name in MODEL_FILES.items():
        dest = MODELS_DIR / local_name
        if dest.exists():
            continue
        if not _download_file(hf_hub_download, hub_name, dest):
            success = False

    if success:
        version_file.write_text(MODEL_VERSION)

    _ensure_samples_downloaded()
    return success


def _ensure_samples_downloaded() -> None:
    """Download curated demo images from HF Hub if missing."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return

    for local_rel, hub_name in SAMPLE_FILES.items():
        dest = _PROJECT_ROOT / local_rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info(f"  Downloading sample {hub_name} ...")
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hub_name,
                repo_type="model",
                local_dir=str(dest.parent),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download saves as the basename of hub_name
            downloaded = dest.parent / Path(hub_name).name
            if downloaded.exists() and downloaded != dest:
                downloaded.rename(dest)
            logger.info(f"  Sample saved to {dest}")
        except Exception as exc:
            logger.warning(f"  Could not download sample {hub_name}: {exc}")
