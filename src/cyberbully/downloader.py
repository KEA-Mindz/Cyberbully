"""Model download and cache utilities for cyberbullying detection."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import sys

logger = logging.getLogger(__name__)

DEFAULT_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1J0xiFBJxbwLREqR6WhPzaxd-PyNA28uQ?usp=sharing"
FALLBACK_DRIVE_FOLDER = "https://drive.google.com/drive/folders/142uQgqE1JsclxDh-DL-ionCB8Flqbf8H?usp=sharing"


def get_default_cache_dir() -> Path:
    """Return default local model cache directory (~/.cache/cyberbully/models)."""
    base = os.environ.get("CYBERBULLY_CACHE_DIR")
    if base:
        cache_path = Path(base)
    else:
        cache_path = Path.home() / ".cache" / "cyberbully" / "models"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def is_model_directory_valid(model_dir: Path | str) -> bool:
    """Check if model directory has required files for inference."""
    path = Path(model_dir)
    if not path.is_dir():
        return False
    
    weights = path / "best_model.pt"
    tokenizer_dir = path / "tokenizer"
    encoder_dir = path / "encoder_config"
    run_config = path / "run_config.json"
    
    if not weights.is_file() or weights.stat().st_size < 1000:
        return False
    if not tokenizer_dir.is_dir() or not (tokenizer_dir / "tokenizer.json").is_file():
        return False
    if not encoder_dir.is_dir() or not (encoder_dir / "config.json").is_file():
        return False
    if not run_config.is_file():
        return False
        
    return True


def download_model(
    target_dir: Path | str | None = None,
    folder_url: str = DEFAULT_DRIVE_FOLDER,
    force: bool = False,
) -> Path:
    """Download model assets from Google Drive into target directory.
    
    Args:
        target_dir: Destination folder path. Defaults to ~/.cache/cyberbully/models/cyberbully_v0.1_run
        folder_url: Google Drive folder share link
        force: If True, re-downloads even if valid directory already exists.
        
    Returns:
        Path to the validated model directory.
    """
    if target_dir is None:
        target_dir = get_default_cache_dir() / "cyberbully_v0.1_run"
    else:
        target_dir = Path(target_dir)

    if not force and is_model_directory_valid(target_dir):
        logger.info(f"Model already exists at: {target_dir}")
        return target_dir

    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download model files automatically. "
            "Please install it with: pip install gdown"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading model checkpoint from Google Drive to: {target_dir}")

    try:
        gdown.download_folder(url=folder_url, output=str(target_dir), quiet=False)
    except Exception as e:
        logger.warning(f"Download from primary folder failed: {e}. Trying fallback link...")
        gdown.download_folder(url=FALLBACK_DRIVE_FOLDER, output=str(target_dir), quiet=False)

    if not is_model_directory_valid(target_dir):
        # Look if gdown created a subfolder inside target_dir
        subdirs = [p for p in target_dir.iterdir() if p.is_dir()]
        for sd in subdirs:
            if is_model_directory_valid(sd):
                logger.info(f"Found model files in subdirectory: {sd}")
                return sd
        raise RuntimeError(
            f"Download completed but valid model structure was not found in: {target_dir}. "
            f"Expected 'best_model.pt', 'tokenizer/', 'encoder_config/', and 'run_config.json'."
        )

    return target_dir
