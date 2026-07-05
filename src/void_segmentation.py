"""Robust and simple void segmentation for BGA solder balls.

Goal:
    Detect ONLY real bright void bubbles inside each solder ball.
    Eliminate hallucinations by removing Hough transforms and complex validations.
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class VoidSegmentationConfig:
    """Configuration class kept for compatibility with main.py."""
    # We keep the attributes so other files importing this don't crash,
    # but the internal logic uses a much safer, simplified path.
    analysis_radius_scale: float = 0.82
    inner_radius_scale: float = 0.68
    hard_rim_radius_scale: float = 0.88
    gaussian_kernel_size: int = 3
    local_background_scale: float = 1.0 # Marit pentru a proteja void-urile mari
    min_void_radius_px: int = 2
    min_void_radius_ratio: float = 0.03
    max_void_radius_ratio: float = 0.28
    min_component_area: int = 10
    min_component_area_ratio: float = 0.0015
    max_component_area_ratio: float = 0.20
    residual_threshold_min_low: float = 12.0 # Contrastul minim clar
    residual_threshold_min_mid: float = 15.0
    residual_threshold_min_high: float = 20.0
    raw_delta_low: float = 5.0
    raw_delta_mid: float = 8.0
    raw_delta_high: float = 11.0
    use_hough: bool = False # FORTAT PE FALSE PENTRU A OPRI HALUCINATIILE
    hough_dp: float = 1.15
    hough_param1: float = 85.0
    hough_param2: float = 12.0
    hough_min_dist_factor: float = 1.25
    large_void_area_ratio: float = 0.012
    min_circularity_small: float = 0.15
    min_circularity_large: float = 0.10
    max_aspect_ratio_small: float = 2.50
    max_aspect_ratio_large: float = 3.50
    max_axis_ratio_small: float = 2.50
    max_axis_ratio_large: float = 3.50
    min_extent_small: float = 0.20
    min_extent_large: float = 0.15
    min_mean_delta_small: float = 4.0
    min_peak_delta_small: float = 6.0
    min_bright_fraction_small: float = 0.15
    min_mean_delta_large: float = 3.0
    min_peak_delta_large: float = 5.0
    min_bright_fraction_large: float = 0.15
    max_outer_fraction_small: float = 0.42
    max_outer_fraction_large: float = 0.55
    max_hard_rim_fraction: float = 0.30
    max_angular_span_degrees: float = 125.0
    duplicate_center_factor: float = 0.85
    duplicate_overlap_ratio: float = 0.35
    draw_radius_scale: float = 1.00
    max_candidates_per_ball: int = 15


def segment_voids(
    roi: np.ndarray,
    local_center: tuple[int, int],
    radius: int,
    config: VoidSegmentationConfig,
) -> np.ndarray:
    """
    FOOLPROOF VOID DETECTION:
    1. Smooth the image.
    2. Create a heavy median background to erase bright voids.
    3. Subtract background from original -> bright spots remain.
    4. Simple Threshold + Contour filtering. No hallucinations.
    """
    if roi.size == 0:
        return np.zeros_like(roi, dtype=np.uint8)

    # Convert to standard format
    raw = roi.copy()
    if raw.dtype != np.uint8:
        raw = np.clip(raw, 0, 255).astype(np.uint8)

    # 1. MASKING - Avoid the dark edges of the ball
    analysis_radius = max(1, int(round(radius * config.analysis_radius_scale)))
    mask = np.zeros_like(raw, dtype=np.uint8)
    cv2.circle(mask, local_center, analysis_radius, 255, thickness=-1)

    # Smooth raw image lightly to kill 1px sensor noise
    smoothed = cv2.GaussianBlur(raw, (3, 3), 0)

    # 2. BACKGROUND ESTIMATION
    # Create a kernel large enough to completely cover and erase huge voids
    k_size = max(11, int(radius * 1.1))
    if k_size % 2 == 0:
        k_size += 1 # Ensure odd

    bg = cv2.medianBlur(smoothed, k_size)

    # 3. SUBTRACTION (Original - Background)
    # Voids will stand out as bright values, normal ball texture will be near 0
    residual = cv2.subtract(smoothed, bg)
    residual = cv2.bitwise_and(residual, mask)

    # 4. THRESHOLDING
    # Any spot that is at least ~12 intensity points brighter than its background is a void candidate
    threshold_value = int(config.residual_threshold_min_low)
    _, binary = cv2.threshold(residual, threshold_value, 255, cv2.THRESH_BINARY)

    # Cleanup tiny 1-2 px noise dots from the binary mask
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)

    # 5. FILTERING
    final_mask = np.zeros_like(raw, dtype=np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignora praful prea mic
        if area < config.min_component_area:
            continue

        # Verifica bounding box pentru a elimina liniile subtiri
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / max(1, min(w, h))

        # Calculați circularitatea
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
        else:
            circularity = 0.0

        # Reguli de acceptare ultra-simple
        is_large = area > (np.pi * (radius**2)) * config.large_void_area_ratio

        if is_large:
            # La void-uri uriase, le permitem sa fie deformate (limite super relaxate)
            if aspect_ratio > config.max_aspect_ratio_large:
                continue
        else:
            # La void-uri mici, respingem zgarieturile/liniile
            if aspect_ratio > config.max_aspect_ratio_small or circularity < config.min_circularity_small:
                continue

        # Desenează void-ul validat (folosim conturul exact, nu un cerc aproximativ)
        cv2.drawContours(final_mask, [cnt], -1, 255, thickness=-1)

    return final_mask