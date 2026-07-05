"""Visualization helpers for BGA ball and void inspection outputs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.ball_detection import BallDetectionDiagnostics, RoiBounds, SolderBall


def save_image(path: Path, image: np.ndarray) -> None:
    """Save an image and fail loudly if OpenCV cannot write it."""
    success = cv2.imwrite(str(path), image)
    if not success:
        raise ValueError(f"Could not write image: {path}")


def draw_bga_roi_debug(
    grayscale: np.ndarray,
    diagnostics: BallDetectionDiagnostics,
) -> np.ndarray:
    """Draw raw pad candidates and the detected BGA ROI."""
    canvas = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    for x, y, radius in diagnostics.raw_candidates:
        cv2.circle(canvas, (x, y), radius, (0, 180, 255), 1)
        cv2.circle(canvas, (x, y), 2, (0, 180, 255), -1)

    if diagnostics.bga_roi is not None:
        roi = diagnostics.bga_roi
        cv2.rectangle(
            canvas,
            (roi.x_min, roi.y_min),
            (roi.x_max, roi.y_max),
            (255, 0, 0),
            3,
        )
    return canvas


def draw_pad_contour_overlay(
    grayscale: np.ndarray,
    balls: list[SolderBall],
    diagnostics: BallDetectionDiagnostics | None = None,
    detection_image: np.ndarray | None = None,
    include_roi: bool = True,
) -> np.ndarray:
    """Draw stable green pad contours for the final BGA slot candidates."""
    canvas = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    pitch = _estimate_visual_pitch(balls)
    fallback_radius = _median_ball_radius(balls)

    if include_roi and diagnostics is not None and diagnostics.bga_roi is not None:
        roi = diagnostics.bga_roi
        cv2.rectangle(
            canvas,
            (roi.x_min, roi.y_min),
            (roi.x_max, roi.y_max),
            (255, 0, 0),
            3,
        )

    for ball in balls:
        if ball.is_estimated:
            radius = max(ball.radius, fallback_radius)
            radius = min(radius, int(round(pitch * 0.36)))
            radius = max(3, int(round(radius * 0.96)))
            cv2.circle(canvas, (ball.center_x, ball.center_y), radius, (0, 120, 0), 2, cv2.LINE_AA)
            continue

        radius = max(ball.radius, fallback_radius)
        radius = min(radius, int(round(pitch * 0.36)))
        radius = max(3, int(round(radius * 0.96)))
        contour = _local_pad_component_contour(
            image=detection_image if detection_image is not None else grayscale,
            ball=ball,
            radius=radius,
            pitch=pitch,
        )
        component_circle = (
            _circle_from_component_contour(contour, ball, radius)
            if contour is not None
            else None
        )
        detection_img = detection_image if detection_image is not None else grayscale
        drew_component = False
        if component_circle is not None:
            center_x, center_y, component_radius = component_circle
            if _has_visual_pad_evidence_at(
                image=detection_img,
                center_x=center_x,
                center_y=center_y,
                radius=component_radius,
            ):
                cv2.circle(
                    canvas,
                    (center_x, center_y),
                    component_radius,
                    (0, 190, 0),
                    2,
                    cv2.LINE_AA,
                )
                drew_component = True

        if not drew_component and _has_visual_pad_evidence_at(
            image=detection_img,
            center_x=ball.center_x,
            center_y=ball.center_y,
            radius=radius,
        ):
            cv2.circle(
                canvas,
                (ball.center_x, ball.center_y),
                radius,
                (0, 190, 0),
                2,
                cv2.LINE_AA,
            )

    return canvas


def _estimate_visual_pitch(balls: list[SolderBall]) -> float:
    if len(balls) < 4:
        return 70.0

    points = np.array(
        [(ball.center_x, ball.center_y) for ball in balls],
        dtype=np.float32,
    )
    nearest_distances: list[float] = []
    for index, point in enumerate(points):
        deltas = points - point
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        distances[index] = np.inf
        valid = distances[(distances >= 25.0) & (distances <= 220.0)]
        if valid.size:
            nearest_distances.append(float(np.min(valid)))

    if not nearest_distances:
        return 70.0
    return float(np.median(nearest_distances))


def _median_ball_radius(balls: list[SolderBall]) -> int:
    if not balls:
        return 28
    return max(4, int(round(float(np.median([ball.radius for ball in balls])))))


def _local_pad_component_contour(
    image: np.ndarray,
    ball: SolderBall,
    radius: int,
    pitch: float,
) -> np.ndarray | None:
    """Extract a local dark pad component instead of blindly drawing a circle."""
    height, width = image.shape[:2]
    search_radius = int(round(max(radius * 1.30, pitch * 0.30)))
    x0 = max(0, ball.center_x - search_radius)
    y0 = max(0, ball.center_y - search_radius)
    x1 = min(width - 1, ball.center_x + search_radius)
    y1 = min(height - 1, ball.center_y + search_radius)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = image[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        return None

    local_x = ball.center_x - x0
    local_y = ball.center_y - y0
    yy, xx = np.indices(crop.shape)
    distances = np.hypot(xx - local_x, yy - local_y)
    local_mask = distances <= max(radius * 1.22, 3.0)
    masked_pixels = crop[local_mask]
    if masked_pixels.size == 0:
        return None

    blurred = cv2.GaussianBlur(crop, (5, 5), 0)
    otsu_threshold, _ = cv2.threshold(
        blurred[local_mask],
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    threshold = min(168.0, max(104.0, float(otsu_threshold) + 10.0))
    dark_mask = ((blurred <= threshold) & local_mask).astype(np.uint8) * 255

    kernel_size = max(3, int(round(radius * 0.10)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    expected_area = float(np.pi * (radius**2))
    best: tuple[float, np.ndarray] | None = None
    max_offset = max(radius * 0.72, pitch * 0.20)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < expected_area * 0.30 or area > expected_area * 1.45:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width <= 0 or box_height <= 0:
            continue

        aspect_ratio = max(box_width, box_height) / max(1, min(box_width, box_height))
        if aspect_ratio > 3.10:
            continue

        extent = area / max(1.0, float(box_width * box_height))
        if extent < 0.30:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        circularity = (4.0 * np.pi * area) / (perimeter**2) if perimeter else 0.0
        if aspect_ratio > 2.25 and circularity < 0.12:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        center_x = float(moments["m10"] / moments["m00"])
        center_y = float(moments["m01"] / moments["m00"])
        offset = float(np.hypot(center_x - local_x, center_y - local_y))
        if offset > max_offset:
            continue

        area_score = 1.0 - min(1.0, abs((area / expected_area) - 0.78) / 0.78)
        offset_score = 1.0 - min(1.0, offset / max(1.0, max_offset))
        shape_score = min(1.0, extent) * 0.5 + min(1.0, circularity) * 0.5
        score = area_score + offset_score + shape_score
        full_contour = contour + np.array([[[x0, y0]]], dtype=contour.dtype)
        if best is None or score > best[0]:
            best = (score, full_contour)

    return None if best is None else best[1]


def _circle_from_component_contour(
    contour: np.ndarray,
    ball: SolderBall,
    fallback_radius: int,
) -> tuple[int, int, int] | None:
    """Convert a clean local component to a stable circle for readable QA output."""
    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    offset = float(np.hypot(center_x - ball.center_x, center_y - ball.center_y))
    if offset > max(5.0, fallback_radius * 0.55):
        return None
    if radius < fallback_radius * 0.64 or radius > fallback_radius * 1.28:
        return None

    visual_radius = int(round(max(fallback_radius * 0.82, min(radius * 1.06, fallback_radius * 1.08))))
    return int(round(center_x)), int(round(center_y)), max(3, visual_radius)


def _has_visual_pad_evidence(
    image: np.ndarray,
    ball: SolderBall,
    radius: int,
) -> bool:
    return _has_visual_pad_evidence_at(
        image=image,
        center_x=ball.center_x,
        center_y=ball.center_y,
        radius=radius,
    )


def _has_visual_pad_evidence_at(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
) -> bool:
    height, width = image.shape[:2]
    x0 = center_x - radius
    y0 = center_y - radius
    x1 = center_x + radius
    y1 = center_y + radius
    if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
        return False

    crop = image[y0 : y1 + 1, x0 : x1 + 1]
    yy, xx = np.indices(crop.shape)
    distances = np.hypot(xx - radius, yy - radius)
    mask = distances <= radius * 0.92
    pixels = crop[mask]
    if pixels.size == 0:
        return False

    mean_intensity = float(np.mean(pixels))
    dark_fraction = float(np.count_nonzero(pixels <= 150)) / float(pixels.size)
    return mean_intensity <= 142.0 and dark_fraction >= 0.58


def _local_pad_contour(
    image: np.ndarray,
    ball: SolderBall,
    pitch: float,
    fallback_radius: int,
) -> np.ndarray | None:
    radius = max(ball.radius, fallback_radius)
    search_radius = int(round(max(radius * 1.75, pitch * 0.52)))
    height, width = image.shape[:2]
    x0 = max(0, ball.center_x - search_radius)
    y0 = max(0, ball.center_y - search_radius)
    x1 = min(width - 1, ball.center_x + search_radius)
    y1 = min(height - 1, ball.center_y + search_radius)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = image[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        return None

    blurred = cv2.GaussianBlur(crop, (5, 5), 0)
    otsu_threshold, otsu_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    fixed_threshold = min(158.0, max(92.0, float(otsu_threshold) + 6.0))
    fixed_mask = (blurred <= fixed_threshold).astype(np.uint8) * 255
    adaptive_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        2,
    )
    binary = cv2.bitwise_or(otsu_mask, fixed_mask)
    binary = cv2.bitwise_and(binary, adaptive_mask)

    kernel_size = max(3, int(round(radius * 0.12)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    expected_area = float(np.pi * (radius**2))
    min_area = expected_area * 0.12
    local_center = np.array([ball.center_x - x0, ball.center_y - y0], dtype=np.float32)
    max_offset = max(radius * 0.48, pitch * 0.23)
    best: tuple[float, np.ndarray] | None = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > expected_area * 1.25:
            continue

        box_x, box_y, box_width, box_height = cv2.boundingRect(contour)
        if box_width <= 0 or box_height <= 0:
            continue

        if (
            box_x <= 1
            or box_y <= 1
            or box_x + box_width >= crop.shape[1] - 2
            or box_y + box_height >= crop.shape[0] - 2
        ):
            edge_penalty = 0.35
        else:
            edge_penalty = 0.0

        aspect_ratio = max(box_width, box_height) / max(1, min(box_width, box_height))
        if aspect_ratio > 1.45:
            continue

        extent = area / max(1.0, float(box_width * box_height))
        if extent < 0.28:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        circularity = (4.0 * np.pi * area) / (perimeter**2) if perimeter else 0.0
        if circularity < 0.30:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        center = np.array(
            [
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            ],
            dtype=np.float32,
        )
        offset = float(np.linalg.norm(center - local_center))
        if offset > max_offset:
            continue

        center_x = int(round(x0 + center[0]))
        center_y = int(round(y0 + center[1]))
        component_radius = max(3, int(round(np.sqrt(area / np.pi))))
        darkness = _pad_darkness_score(image, center_x, center_y, component_radius)
        if darkness < 0.58:
            continue

        area_ratio = area / expected_area
        area_score = 1.0 - min(1.0, abs(area_ratio - 0.62) / 0.75)
        distance_score = 1.0 - min(1.0, offset / max_offset)
        shape_score = float(np.clip((extent + circularity) / 2.0, 0.0, 1.0))
        score = (
            darkness * 1.6
            + area_score * 0.7
            + distance_score * 0.7
            + shape_score * 0.5
            - edge_penalty
        )

        shifted = contour.copy()
        shifted[:, 0, 0] += x0
        shifted[:, 0, 1] += y0
        if best is None or score > best[0]:
            best = (score, shifted)

    return None if best is None else best[1]


def _pad_darkness_score(image: np.ndarray, x: int, y: int, radius: int) -> float:
    height, width = image.shape[:2]
    radius = max(3, int(radius))
    if x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height:
        return 0.0

    crop = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
    yy, xx = np.indices(crop.shape)
    distances = np.hypot(xx - radius, yy - radius)
    mask = distances <= radius * 0.86
    pixels = crop[mask]
    if pixels.size == 0:
        return 0.0

    mean_score = np.clip((188.0 - float(np.mean(pixels))) / 95.0, 0.0, 1.0)
    dark_fraction = float(np.count_nonzero(pixels <= 152)) / float(pixels.size)
    fraction_score = np.clip(dark_fraction / 0.42, 0.0, 1.0)
    return float((mean_score + fraction_score) / 2.0)


def _draw_contour_or_ellipse(canvas: np.ndarray, contour: np.ndarray) -> None:
    cv2.drawContours(canvas, [contour], -1, (0, 190, 0), 2, cv2.LINE_AA)


def annotate_detected_balls(
    grayscale: np.ndarray,
    balls: list[SolderBall],
    rows: list[dict[str, object]],
    warning_threshold: float,
) -> np.ndarray:
    """Draw detected solder balls and void percentages on a color image."""
    annotated = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    ratio_by_ball_id = {
        int(row["ball_id"]): float(row["void_ratio_percent"])
        for row in rows
    }

    for ball in balls:
        ratio = ratio_by_ball_id.get(ball.ball_id, 0.0)

        # ROSU daca pica testul, VERDE daca e sub limita
        color = (0, 0, 255) if ratio >= warning_threshold else (0, 180, 0)
        center = (ball.center_x, ball.center_y)

        cv2.circle(annotated, center, ball.radius, color, thickness=2)

        # Acum afisam si ID-ul bilei SI PROCENTUL de void
        text = f"#{ball.ball_id}: {ratio:.1f}%"

        # Mutam textul deasupra bilei ca sa se vada clar
        cv2.putText(
            annotated,
            text,
            (ball.center_x - 20, ball.center_y - ball.radius - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def create_void_mask_preview(
    grayscale: np.ndarray,
    balls: list[SolderBall],
    mask_entries: list[tuple[RoiBounds, np.ndarray]],
) -> np.ndarray:
    """Create a full-image preview with void masks overlaid in red."""
    preview = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    full_mask = np.zeros_like(grayscale, dtype=np.uint8)

    for bounds, mask in mask_entries:
        target = full_mask[bounds.y_min:bounds.y_max, bounds.x_min:bounds.x_max]
        if target.shape == mask.shape:
            full_mask[bounds.y_min:bounds.y_max, bounds.x_min:bounds.x_max] = (
                cv2.bitwise_or(target, mask)
            )

    red_overlay = np.zeros_like(preview, dtype=np.uint8)
    red_overlay[:, :, 2] = full_mask
    preview = cv2.addWeighted(preview, 1.0, red_overlay, 0.65, 0)

    for ball in balls:
        cv2.circle(
            preview,
            (ball.center_x, ball.center_y),
            ball.radius,
            (0, 180, 0),
            thickness=1,
        )

    return preview