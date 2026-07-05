"""Connected component metrics for segmented void masks.

This module performs final sanity filtering and computes per-ball metrics.
The segmentation stage should already return only compact void-like regions;
this file protects the report from remaining tiny dots, rim arcs and stripes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import cv2
import numpy as np


@dataclass(frozen=True)
class ComponentFilterConfig:
    """Filtering parameters for final void connected components."""

    # Aliniat cu segmentarea noastra (nimic nu se mai pierde pe drum)
    min_void_area: int = 10
    min_void_area_ratio: float = 0.0015
    min_outer_void_area_ratio: float = 0.0040
    max_component_area_ratio: float = 0.24

    # Normal voids.
    min_circularity: float = 0.10             # Relaxat pentru voiduri asimetrice
    max_aspect_ratio: float = 3.20            # Permitem void-uri alungite/lipite
    max_principal_axis_ratio: float = 3.40
    min_extent: float = 0.10

    # Large voids may be slightly irregular.
    large_void_area_ratio: float = 0.012
    large_void_min_circularity: float = 0.05
    large_void_max_aspect_ratio: float = 3.50
    large_void_max_principal_axis_ratio: float = 3.85
    large_void_min_extent: float = 0.08

    # Rim/line rejection.
    rim_start_radius_ratio: float = 0.70
    hard_rim_radius_ratio: float = 0.84
    max_outer_fraction_normal: float = 0.50
    max_outer_fraction_large: float = 0.56
    max_hard_rim_fraction: float = 0.40
    max_angular_span_degrees: float = 155.0


@dataclass(frozen=True)
class VoidMetrics:
    """Void statistics for one solder ball."""

    void_count: int
    total_void_area: int
    largest_void_area: int


@dataclass(frozen=True)
class _ComponentFeatures:
    """Geometry features for one final connected component."""

    area: int
    area_ratio: float
    aspect_ratio: float
    extent: float
    circularity: float
    principal_axis_ratio: float
    outer_fraction: float
    hard_rim_fraction: float
    angular_span_degrees: float


def analyze_void_components(
    mask: np.ndarray,
    ball_area: float,
    config: ComponentFilterConfig,
    roi: np.ndarray | None = None,
    rejection_log: list[str] | None = None,
) -> tuple[VoidMetrics, np.ndarray]:
    """Measure connected void components and return a final cleaned mask."""
    if mask.size == 0:
        return VoidMetrics(0, 0, 0), np.zeros_like(mask, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    clean_mask = np.zeros_like(mask, dtype=np.uint8)
    accepted_areas: list[int] = []

    height, width = mask.shape[:2]
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    estimated_radius = max(1.0, min(width, height) / 2.0)

    yy, xx = np.indices(mask.shape)
    distances = np.hypot(xx - center_x, yy - center_y)

    min_area = max(
        config.min_void_area,
        int(round(ball_area * config.min_void_area_ratio)),
    )

    for label_id in range(1, num_labels):
        component = labels == label_id
        component_mask = np.where(component, 255, 0).astype(np.uint8)

        features = _component_features(
            component=component,
            component_mask=component_mask,
            stats=stats[label_id],
            ball_area=ball_area,
            distances=distances,
            estimated_radius=estimated_radius,
            center_x=center_x,
            center_y=center_y,
        )

        if features.area < min_area:
            _log_rejection(rejection_log, f"area={features.area}<min_area={min_area}")
            continue

        if features.area_ratio > config.max_component_area_ratio:
            _log_rejection(
                rejection_log,
                f"area_ratio={features.area_ratio:.3f}>max={config.max_component_area_ratio}",
            )
            continue

        if _looks_like_arc_or_line(features, config):
            _log_rejection(
                rejection_log,
                "arc_or_line("
                f"area_ratio={features.area_ratio:.3f}, "
                f"aspect={features.aspect_ratio:.2f}, "
                f"principal={features.principal_axis_ratio:.2f}, "
                f"circ={features.circularity:.2f}, "
                f"outer={features.outer_fraction:.2f}, "
                f"hard={features.hard_rim_fraction:.2f}, "
                f"span={features.angular_span_degrees:.1f})",
            )
            continue

        if not _passes_shape_filter(features, config):
            _log_rejection(
                rejection_log,
                "shape("
                f"area_ratio={features.area_ratio:.3f}, "
                f"aspect={features.aspect_ratio:.2f}, "
                f"principal={features.principal_axis_ratio:.2f}, "
                f"extent={features.extent:.2f}, "
                f"circ={features.circularity:.2f})",
            )
            continue

        # Small components near the outer ring are usually rim texture.
        if features.outer_fraction > 0.50:
            outer_min_area = max(
                min_area,
                int(round(ball_area * config.min_outer_void_area_ratio)),
            )
            if features.area < outer_min_area:
                _log_rejection(
                    rejection_log,
                    f"outer_small_area={features.area}<outer_min={outer_min_area}",
                )
                continue

        clean_mask[component] = 255
        accepted_areas.append(features.area)

    total_area = int(sum(accepted_areas))
    largest_area = int(max(accepted_areas)) if accepted_areas else 0

    metrics = VoidMetrics(
        void_count=len(accepted_areas),
        total_void_area=total_area,
        largest_void_area=largest_area,
    )

    return metrics, clean_mask


def _component_features(
    component: np.ndarray,
    component_mask: np.ndarray,
    stats: np.ndarray,
    ball_area: float,
    distances: np.ndarray,
    estimated_radius: float,
    center_x: float,
    center_y: float,
) -> _ComponentFeatures:
    """Compute geometry features for one connected component."""
    area = int(stats[cv2.CC_STAT_AREA])
    area_ratio = area / max(1.0, ball_area)

    box_width = int(stats[cv2.CC_STAT_WIDTH])
    box_height = int(stats[cv2.CC_STAT_HEIGHT])
    aspect_ratio = max(box_width, box_height) / max(1, min(box_width, box_height))
    extent = area / max(1, box_width * box_height)
    circularity = _component_circularity(component_mask, area)
    principal_axis_ratio = _principal_axis_ratio(component_mask)

    component_distances = distances[component]
    if component_distances.size == 0:
        outer_fraction = 1.0
        hard_rim_fraction = 1.0
    else:
        outer_fraction = float(
            np.count_nonzero(component_distances >= estimated_radius * 0.70),
        ) / float(component_distances.size)
        hard_rim_fraction = float(
            np.count_nonzero(component_distances >= estimated_radius * 0.84),
        ) / float(component_distances.size)

    angular_span = _angular_span_degrees(component, center_x, center_y)

    return _ComponentFeatures(
        area=area,
        area_ratio=area_ratio,
        aspect_ratio=aspect_ratio,
        extent=extent,
        circularity=circularity,
        principal_axis_ratio=principal_axis_ratio,
        outer_fraction=outer_fraction,
        hard_rim_fraction=hard_rim_fraction,
        angular_span_degrees=angular_span,
    )


def _passes_shape_filter(
    features: _ComponentFeatures,
    config: ComponentFilterConfig,
) -> bool:
    """Apply compactness checks while allowing real large voids."""
    is_large = features.area_ratio >= config.large_void_area_ratio

    if is_large:
        if features.aspect_ratio > config.large_void_max_aspect_ratio:
            return False
        if features.principal_axis_ratio > config.large_void_max_principal_axis_ratio:
            return False
        if features.extent < config.large_void_min_extent:
            return False
        if features.circularity < config.large_void_min_circularity:
            return False
        return True

    if features.aspect_ratio > config.max_aspect_ratio:
        return False
    if features.principal_axis_ratio > config.max_principal_axis_ratio:
        return False
    if features.extent < config.min_extent:
        return False
    if features.circularity < config.min_circularity:
        return False
    return True


def _looks_like_arc_or_line(
    features: _ComponentFeatures,
    config: ComponentFilterConfig,
) -> bool:
    """Reject non-void components: rim arcs, stripes and elongated traces."""
    is_large = features.area_ratio >= config.large_void_area_ratio
    outer_limit = (
        config.max_outer_fraction_large
        if is_large
        else config.max_outer_fraction_normal
    )

    if features.aspect_ratio > 2.30 and features.circularity < 0.36:
        return True

    if features.principal_axis_ratio > 2.65 and features.circularity < 0.36:
        return True

    if features.hard_rim_fraction > config.max_hard_rim_fraction:
        if features.circularity < 0.42:
            return True
        if features.aspect_ratio > 1.90:
            return True

    if features.outer_fraction > outer_limit and features.circularity < 0.40:
        return True

    if (
        features.angular_span_degrees > config.max_angular_span_degrees
        and features.outer_fraction > 0.25
        and features.circularity < 0.48
    ):
        return True

    if (
        features.area_ratio > 0.025
        and features.outer_fraction > 0.42
        and features.circularity < 0.46
    ):
        return True

    return False


def _component_circularity(component_mask: np.ndarray, area: int) -> float:
    """Estimate component circularity from contour perimeter."""
    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return 0.0

    perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), True))
    if perimeter <= 0.0:
        return 0.0

    return float((4.0 * pi * area) / (perimeter**2))


def _principal_axis_ratio(component_mask: np.ndarray) -> float:
    """Return the oriented elongation ratio of one connected component."""
    y_coords, x_coords = np.nonzero(component_mask)
    if len(x_coords) < 3:
        return 1.0

    coordinates = np.column_stack((x_coords, y_coords)).astype(np.float64)
    covariance = np.cov(coordinates, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)

    smallest = max(float(eigenvalues[0]), 1e-6)
    largest = max(float(eigenvalues[-1]), smallest)
    return float(np.sqrt(largest / smallest))


def _angular_span_degrees(
    component: np.ndarray,
    center_x: float,
    center_y: float,
) -> float:
    """Estimate angular span around the solder ball center."""
    y_coords, x_coords = np.nonzero(component)
    if len(x_coords) < 2:
        return 0.0

    angles = np.degrees(np.arctan2(y_coords - center_y, x_coords - center_x))
    angles = np.mod(angles, 360.0)
    angles = np.sort(angles)
    gaps = np.diff(np.concatenate([angles, angles[:1] + 360.0]))
    largest_gap = float(np.max(gaps))
    return float(360.0 - largest_gap)


def _log_rejection(rejection_log: list[str] | None, message: str) -> None:
    """Append rejection reason if debugging is enabled."""
    if rejection_log is not None:
        rejection_log.append(message)