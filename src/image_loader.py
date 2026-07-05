"""Image loading helpers for the BGA inspection pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def load_grayscale(image_path: Path) -> np.ndarray:
    """Load an image as an 8-bit grayscale array."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image as grayscale: {image_path}")
    return image


def list_image_files(input_dir: Path) -> list[Path]:
    """Return supported image files from a directory tree."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
