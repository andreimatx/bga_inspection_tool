"""Preprocessing operations for X-ray BGA images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters for contrast enhancement and noise reduction."""

    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    median_kernel_size: int = 5


def apply_clahe(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Enhance local contrast using CLAHE."""
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )
    return clahe.apply(image)


def apply_median_filter(
    image: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Reduce local noise with a median filter."""
    return cv2.medianBlur(image, config.median_kernel_size)


def preprocess_image(
    image: np.ndarray,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply CLAHE followed by median filtering."""
    clahe_image = apply_clahe(image, config)
    denoised = apply_median_filter(clahe_image, config)
    return clahe_image, denoised
