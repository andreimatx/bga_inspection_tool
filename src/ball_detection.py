"""BGA solder ball detection using the Hough Circle Transform."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import pi
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class HoughCircleConfig:
    """Parameters for Hough-based solder ball detection."""

    dp: float = 1.2
    min_dist: float = 55.0
    param1: float = 110.0
    param2: float = 28.0
    min_radius: int = 30
    max_radius: int = 46
    max_circle_mean_intensity: float = 115.0
    max_core_mean_intensity: float = 112.0
    min_dark_pixel_fraction: float = 0.52
    min_outer_dark_pixel_fraction: float = 0.56
    dark_pixel_intensity_threshold: int = 115
    core_radius_scale: float = 0.55
    outer_ring_inner_radius_scale: float = 0.62
    duplicate_distance_factor: float = 0.55
    alignment_tolerance: int = 26
    min_aligned_neighbors: int = 2
    min_neighbor_pitch: int = 55
    max_neighbor_pitch: int = 320
    min_layout_keep_ratio: float = 0.65
    reject_trace_like_components: bool = True
    dark_component_seed_radius_scale: float = 0.25
    dark_component_close_kernel_size: int = 3
    max_trace_component_aspect_ratio: float = 1.32
    max_trace_component_area_ratio: float = 0.72
    normalize_small_pad_radii: bool = True
    normalized_radius_median_ratio: float = 1.0
    normalized_radius_min_count: int = 20
    normalized_radius_max_boost: int = 11
    recover_grid_gaps: bool = True
    grid_gap_min_count: int = 20
    grid_gap_pitch_tolerance_ratio: float = 0.28
    grid_gap_duplicate_distance: float = 34.0
    grid_gap_max_mean_intensity: float = 90.0
    grid_gap_max_core_mean_intensity: float = 95.0
    grid_gap_min_dark_pixel_fraction: float = 0.70
    grid_gap_min_outer_dark_pixel_fraction: float = 0.56
    grid_gap_max_component_aspect_ratio: float = 1.26
    grid_gap_min_component_area_ratio: float = 0.66
    regularize_grid_centers: bool = True
    grid_regularization_min_axis_count: int = 4
    grid_regularization_min_fill_ratio: float = 0.55
    grid_regularization_max_cluster_std: float = 9.0
    grid_regularization_max_pitch_cv: float = 0.08
    grid_regularization_snap_tolerance: float = 32.0
    refine_boundary: bool = True
    boundary_search_scale: float = 1.45
    boundary_min_scale: float = 0.72
    boundary_angle_count: int = 128
    boundary_radius_percentile: float = 64.0
    boundary_min_gradient: float = 12.0
    boundary_window_size: int = 3
    boundary_min_valid_fraction: float = 0.18
    boundary_small_radius_threshold: int = 34
    boundary_medium_radius_threshold: int = 39
    boundary_small_radius_growth_scale: float = 1.42
    boundary_medium_radius_growth_scale: float = 1.30
    boundary_large_radius_growth_scale: float = 1.16
    normalize_contour_radii: bool = True
    contour_radius_min_count: int = 40
    contour_radius_floor_percentile: float = 90.0
    contour_radius_max_boost: int = 5
    contour_radius_max_pitch_ratio: float = 0.34
    require_complete_circle: bool = True
    use_bga_grid_roi: bool = True
    expected_grid_rows: int = 16
    expected_grid_cols: int = 16
    roi_detection_max_dimension: int = 700
    roi_candidate_min_dist_scaled: float = 7.0
    roi_candidate_param2: float = 15.0
    roi_candidate_min_radius_scaled: int = 4
    roi_candidate_max_radius_scaled: int = 18
    roi_candidate_dark_threshold: int = 135
    roi_candidate_max_mean_intensity: float = 150.0
    roi_candidate_min_dark_fraction: float = 0.32
    grid_search_min_pitch: float = 45.0
    grid_search_max_pitch: float = 220.0
    grid_pitch_histogram_bin: float = 4.0
    grid_pitch_keep_count: int = 4
    grid_axis_candidate_keep_count: int = 10
    grid_line_tolerance_ratio: float = 0.22
    grid_slot_tolerance_ratio: float = 0.34
    grid_roi_margin_pitch_ratio: float = 0.75
    grid_min_occupied_slots: int = 120
    grid_radius_pitch_ratio: float = 0.30
    grid_radius_min: int = 8
    grid_refine_min_axis_assignments: int = 3
    roi_blob_detection_max_dimension: int = 1400
    roi_blob_min_area: float = 20.0
    roi_blob_max_area: float = 3500.0
    roi_blob_max_aspect_ratio: float = 2.0
    roi_blob_min_circularity: float = 0.25
    roi_blob_min_extent: float = 0.35
    roi_blob_radius_min_ratio: float = 0.55
    roi_blob_radius_max_ratio: float = 1.75
    roi_blob_pitch_min_radius_ratio: float = 2.55
    roi_blob_pitch_max_radius_ratio: float = 6.2
    roi_blob_min_component_candidates: int = 70
    roi_blob_quantile: float = 0.01
    roi_blob_margin_pitch_ratio: float = 1.25
    hough_preferred_min_detected_ratio: float = 0.95
    roi_candidate_min_score: float = 0.22
    roi_neighbor_tolerance_ratio: float = 0.30
    roi_neighbor_min_pitch_ratio: float = 0.45
    roi_neighbor_max_pitch_ratio: float = 1.45
    roi_pad_radius_min_pitch_ratio: float = 0.28
    roi_pad_radius_seed_pitch_ratio: float = 0.30
    roi_pad_radius_max_pitch_ratio: float = 0.38
    roi_component_refine_min_ratio: float = 0.58
    roi_component_margin_pitch_ratio: float = 0.60
    roi_component_expected_span_min_ratio: float = 0.85
    roi_component_keep_full_ratio: float = 0.94
    roi_local_hough_min_detected_ratio: float = 0.98
    roi_local_hough_broad_min_detected_ratio: float = 0.82
    roi_dense_row_tolerance_ratio: float = 0.42
    roi_dense_row_min_fraction: float = 0.38
    roi_dense_row_min_count: int = 4
    roi_dense_row_margin_pitch_ratio: float = 0.62
    recover_grid_evidence_candidates: bool = True
    grid_evidence_min_score: float = 0.65
    grid_evidence_max_recovered_ratio: float = 0.25
    prefer_roi_component_candidates: bool = True
    roi_component_candidate_min_result_ratio: float = 0.86
    roi_component_candidate_replace_ratio: float = 0.90
    roi_component_pad_min_area_ratio: float = 0.04
    roi_component_pad_max_area_ratio: float = 0.75
    roi_component_pad_max_aspect_ratio: float = 3.8
    roi_component_pad_min_extent: float = 0.35
    roi_component_pad_min_circularity: float = 0.12
    roi_component_axis_cluster_tolerance_ratio: float = 0.48
    roi_component_axis_assign_tolerance_ratio: float = 0.52
    roi_component_grid_evidence_min_score: float = 0.62
    final_grid_regularization: bool = True
    final_grid_min_assignment_ratio: float = 0.58
    final_grid_candidate_min_score: float = 0.22
    final_grid_assign_tolerance_ratio: float = 0.48
    final_grid_recover_candidates: bool = True
    final_grid_recovery_min_score: float = 0.72
    final_grid_axis_keep_count: int = 12
    final_grid_component_fit_min_ratio: float = 0.70
    final_grid_component_assign_min_ratio: float = 0.70
    final_grid_axis_min_gap_ratio: float = 0.62
    final_grid_axis_max_gap_ratio: float = 1.55
    final_grid_refine_min_axis_fraction: float = 0.18
    final_grid_slot_recovery_zoom_target_radius: float = 52.0
    final_grid_slot_recovery_max_zoom: float = 3.0
    final_grid_slot_recovery_search_pitch_ratio: float = 0.44
    final_grid_slot_recovery_max_offset_ratio: float = 0.34
    final_grid_slot_recovery_min_area_ratio: float = 0.12
    final_grid_slot_recovery_max_area_ratio: float = 1.10
    final_grid_use_evidence_weighted_assignment: bool = True
    final_grid_assignment_min_evidence: float = 0.34
    final_grid_assignment_edge_min_evidence: float = 0.34
    final_grid_assignment_distance_weight: float = 0.85
    final_grid_assignment_candidate_weight: float = 0.35
    snap_final_candidates_to_grid: bool = False
    use_affine_final_grid: bool = False
    use_local_component_final_snap: bool = False
    use_post_refine_local_snap: bool = False
    # Oblique X-ray views foreshorten one axis (pitch_x != pitch_y) and make
    # every ball drift slightly against the fitted grid, so the per-ball
    # local snap is enabled automatically for such views. Straight views
    # (pitch ratio ~1.0) keep their exact grid positions.
    auto_post_snap_anisotropic_grids: bool = True
    auto_affine_anisotropic_grids: bool = True
    post_snap_min_pitch_anisotropy: float = 1.12
    bga_roi_bottom_shrink_pitch: float = 0.0
    bga_roi_top_shrink_pitch: float = 0.0   # positive=shrink top down, negative=expand top up
    bga_roi_left_shrink_pitch: float = 0.0  # shrink left edge inward
    bga_roi_right_shrink_pitch: float = 0.0 # shrink right edge inward
    void_evidence_min_area_ratio: float = 0.0025
    void_evidence_max_area_ratio: float = 0.28
    void_evidence_min_intensity_delta: float = 18.0


@dataclass(frozen=True)
class SolderBall:
    """Detected solder ball geometry."""

    ball_id: int
    center_x: int
    center_y: int
    radius: int
    confidence: float = 1.0
    is_estimated: bool = False

    @property
    def diameter(self) -> int:
        """Return the ball diameter in pixels."""
        return self.radius * 2

    @property
    def area(self) -> float:
        """Return the ideal circular ball area in pixels."""
        return pi * (self.radius**2)


@dataclass(frozen=True)
class RoiBounds:
    """ROI bounds in full image coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(frozen=True)
class MissingGridPosition:
    """Expected BGA grid slot without direct local candidate confirmation."""

    row: int
    column: int
    center_x: int
    center_y: int
    reason: str


@dataclass(frozen=True)
class BallDetectionDiagnostics:
    """Intermediate data for explainable BGA ROI/grid debugging."""

    raw_candidates: tuple[tuple[int, int, int], ...]
    bga_roi: RoiBounds | None
    filtered_grid: tuple[tuple[int, int, int], ...]
    rejected_candidates: tuple[tuple[int, int, int], ...]
    missing_grid_positions: tuple[MissingGridPosition, ...]
    occupied_grid_slots: int
    expected_grid_slots: int


@dataclass(frozen=True)
class BallDetectionResult:
    """Detected balls plus optional diagnostic information."""

    balls: list[SolderBall]
    diagnostics: BallDetectionDiagnostics


@dataclass(frozen=True)
class _BgaGridFit:
    """Regular 2D grid fitted over BGA pad candidates."""

    score: float
    occupied: int
    pitch_x: float
    pitch_y: float
    x_positions: tuple[float, ...]
    y_positions: tuple[float, ...]


@dataclass(frozen=True)
class _PadCandidate:
    """Direct local evidence for one possible solder pad."""

    x: int
    y: int
    radius: int
    score: float
    source: str

    @property
    def circle(self) -> tuple[int, int, int]:
        """Return the candidate as a simple circle tuple."""
        return (self.x, self.y, self.radius)


@dataclass(frozen=True)
class _BgaRoiEstimate:
    """BGA package ROI estimated from a dense regular pad component."""

    roi: RoiBounds
    pitch: float
    support_count: int
    source: str


def detect_solder_balls(
    image: np.ndarray,
    config: HoughCircleConfig,
) -> list[SolderBall]:
    """Detect BGA solder balls, preferring the fitted 16x16 package grid."""
    if config.use_bga_grid_roi:
        result = detect_solder_balls_with_diagnostics(image, config)
        if result.balls:
            return result.balls

    return _detect_hough_solder_balls(image, config)


def detect_solder_balls_with_diagnostics(
    image: np.ndarray,
    config: HoughCircleConfig,
) -> BallDetectionResult:
    """Detect the BGA ROI first, then return only directly observed pads."""
    expected_slots = config.expected_grid_rows * config.expected_grid_cols
    blob_candidates = _detect_blob_pad_candidates(image, config)
    hough_balls = _detect_hough_solder_balls(image, config)
    hough_candidates = _pad_candidates_from_balls(hough_balls)
    raw_candidates = _merge_pad_candidates(
        [*blob_candidates, *hough_candidates],
        config,
    )
    final_pitch: float | None = None

    if len(hough_balls) >= int(expected_slots * config.hough_preferred_min_detected_ratio):
        roi = _roi_from_candidate_bounds(
            image=image,
            candidates=[candidate.circle for candidate in hough_candidates],
            margin=float(np.median([ball.radius for ball in hough_balls])),
        )
        final_candidates = _merge_pad_candidates(hough_candidates, config)
        final_pitch = _estimate_pitch_from_candidates(final_candidates, config)
        missing_positions: list[MissingGridPosition] = []
        _occupied_slots = min(len(final_candidates), expected_slots)
    else:
        roi_estimate = _estimate_bga_roi_from_blob_candidates(
            candidates=blob_candidates,
            image_shape=image.shape[:2],
            config=config,
        )
        if roi_estimate is None:
            final_candidates = _merge_pad_candidates(hough_candidates, config)
            roi = (
                _roi_from_candidate_bounds(
                    image=image,
                    candidates=[candidate.circle for candidate in final_candidates],
                    margin=0.0,
                )
                if final_candidates
                else None
            )
            final_pitch = _estimate_pitch_from_candidates(final_candidates, config)
            missing_positions = []
            _occupied_slots = len(final_candidates)
        else:
            roi = roi_estimate.roi
            final_pitch = roi_estimate.pitch
            large_roi_candidates = _large_roi_candidates(
                candidates=raw_candidates,
                roi=roi,
            )
            base_raw_candidates = raw_candidates
            fallback_raw_candidates: list[_PadCandidate] | None = None
            if len(large_roi_candidates) < int(
                expected_slots * config.roi_local_hough_min_detected_ratio,
            ):
                local_hough_candidates = _detect_roi_hough_pad_candidates(
                    image=image,
                    roi_estimate=roi_estimate,
                    config=config,
                )
                if len(large_roi_candidates) < int(
                    expected_slots * config.roi_local_hough_broad_min_detected_ratio,
                ):
                    roi_hough_candidates = local_hough_candidates
                else:
                    max_anchor_distance = roi_estimate.pitch * 0.30
                    roi_hough_candidates = [
                        candidate
                        for candidate in local_hough_candidates
                        if _is_near_anchor_candidate(
                            candidate=candidate,
                            anchors=large_roi_candidates,
                            max_distance=max_anchor_distance,
                        )
                    ]
                    fallback_raw_candidates = _merge_pad_candidates(
                        [*base_raw_candidates, *local_hough_candidates],
                        config,
                    )
                raw_candidates = _merge_pad_candidates(
                    [*base_raw_candidates, *roi_hough_candidates],
                    config,
                )
            final_candidates, missing_positions, _occupied_slots = (
                _filter_bga_roi_pad_candidates(
                    image=image,
                    candidates=raw_candidates,
                    roi_estimate=roi_estimate,
                    config=config,
                )
            )
            if (
                fallback_raw_candidates is not None
                and len(final_candidates) < expected_slots
            ):
                fallback_candidates, fallback_missing, fallback_occupied_slots = (
                    _filter_bga_roi_pad_candidates(
                        image=image,
                        candidates=fallback_raw_candidates,
                        roi_estimate=roi_estimate,
                        config=config,
                    )
                )
                if len(fallback_candidates) > len(final_candidates):
                    raw_candidates = fallback_raw_candidates
                    final_candidates = fallback_candidates
                    missing_positions = fallback_missing
                    _occupied_slots = fallback_occupied_slots

    refined_roi = _refine_roi_from_dominant_component(
        candidates=final_candidates,
        initial_roi=roi,
        image_shape=image.shape[:2],
        pitch=final_pitch,
        config=config,
    )
    if refined_roi is not None:
        roi = refined_roi
        final_candidates = [
            candidate
            for candidate in final_candidates
            if _roi_contains_point(roi, candidate.x, candidate.y)
        ]

    # Trim ROI bottom edge for images where detection overshoots the actual BGA boundary.
    if config.bga_roi_bottom_shrink_pitch > 0.0 and roi is not None and final_pitch is not None:
        shrink_px = int(round(config.bga_roi_bottom_shrink_pitch * final_pitch))
        roi = RoiBounds(
            x_min=roi.x_min,
            y_min=roi.y_min,
            x_max=roi.x_max,
            y_max=max(roi.y_min + int(final_pitch), roi.y_max - shrink_px),
        )
        final_candidates = [
            c for c in final_candidates if _roi_contains_point(roi, c.x, c.y)
        ]

    # Top shrink (positive) or expand (negative — recovers raw candidates above old ROI top).
    if config.bga_roi_top_shrink_pitch != 0.0 and roi is not None and final_pitch is not None:
        shift_px = int(round(config.bga_roi_top_shrink_pitch * final_pitch))
        new_y_min = max(0, roi.y_min + shift_px)
        if shift_px < 0 and new_y_min < roi.y_min:
            existing_xy = {(c.x, c.y) for c in final_candidates}
            extra = [
                c for c in raw_candidates
                if new_y_min <= c.y < roi.y_min
                and roi.x_min <= c.x <= roi.x_max
                and (c.x, c.y) not in existing_xy
            ]
            if extra:
                final_candidates = [*final_candidates, *extra]
        elif shift_px > 0:
            final_candidates = [c for c in final_candidates if c.y >= new_y_min]
        roi = RoiBounds(x_min=roi.x_min, y_min=new_y_min, x_max=roi.x_max, y_max=roi.y_max)

    # Left shrink: removes candidates in non-BGA area at left edge.
    if config.bga_roi_left_shrink_pitch > 0.0 and roi is not None and final_pitch is not None:
        shrink_px = int(round(config.bga_roi_left_shrink_pitch * final_pitch))
        new_x_min = min(roi.x_max - int(final_pitch), roi.x_min + shrink_px)
        roi = RoiBounds(x_min=new_x_min, y_min=roi.y_min, x_max=roi.x_max, y_max=roi.y_max)
        final_candidates = [c for c in final_candidates if _roi_contains_point(roi, c.x, c.y)]

    # Right shrink: removes candidates in non-BGA area at right edge.
    if config.bga_roi_right_shrink_pitch > 0.0 and roi is not None and final_pitch is not None:
        shrink_px = int(round(config.bga_roi_right_shrink_pitch * final_pitch))
        new_x_max = max(roi.x_min + int(final_pitch), roi.x_max - shrink_px)
        roi = RoiBounds(x_min=roi.x_min, y_min=roi.y_min, x_max=new_x_max, y_max=roi.y_max)
        final_candidates = [c for c in final_candidates if _roi_contains_point(roi, c.x, c.y)]

    if (
        config.prefer_roi_component_candidates
        and roi is not None
        and final_pitch is not None
        and len(final_candidates)
        < int(expected_slots * config.hough_preferred_min_detected_ratio)
    ):
        component_candidates = _detect_roi_component_pad_candidates(
            image=image,
            roi=roi,
            pitch=final_pitch,
            config=config,
        )
        if _should_prefer_roi_component_candidates(
            current_candidates=final_candidates,
            component_candidates=component_candidates,
            expected_slots=expected_slots,
            config=config,
        ):
            component_candidates, missing_positions = (
                _regularize_component_candidates_on_axis_grid(
                    image=image,
                    candidates=component_candidates,
                    pitch=final_pitch,
                    config=config,
                )
            )
            raw_candidates = _merge_pad_candidates(
                [*raw_candidates, *component_candidates],
                config,
            )
            final_candidates = component_candidates

    final_grid: _BgaGridFit | None = None
    if (
        config.final_grid_regularization
        and roi is not None
        and final_pitch is not None
    ):
        final_candidates, missing_positions, final_pitch, final_grid = (
            _regularize_final_candidates_on_roi_grid(
                image=image,
                candidates=_merge_pad_candidates(
                    [*raw_candidates, *final_candidates],
                    config,
                ),
                roi=roi,
                pitch=final_pitch,
                config=config,
            )
        )

    balls = _balls_from_real_candidates(
        image=image,
        candidates=final_candidates,
        pitch=final_pitch,
        config=config,
    )

    grid_is_anisotropic = False
    if (
        final_grid is not None
        and final_grid.pitch_x > 0
        and final_grid.pitch_y > 0
    ):
        pitch_anisotropy = max(final_grid.pitch_x, final_grid.pitch_y) / min(
            final_grid.pitch_x,
            final_grid.pitch_y,
        )
        grid_is_anisotropic = (
            pitch_anisotropy >= config.post_snap_min_pitch_anisotropy
        )

    # Fill grid slots not covered by any detected ball (nearest-slot assignment)
    if final_grid is not None and final_pitch is not None and len(balls) < expected_slots:
        grid_xs = final_grid.x_positions
        grid_ys = final_grid.y_positions
        occupied_slots: set[tuple[int, int]] = set()
        fill_centers: list[tuple[int, int]] | None = None
        if grid_is_anisotropic:
            # Oblique views shear ball positions away from the axis-aligned
            # slot centers, so nearest-slot occupancy mapping is unreliable.
            # The affine/homography stage already reports each unconfirmed
            # slot with its projected coordinates — fill exactly those, so
            # every physical pad keeps its own marker even when neighbouring
            # evidence overlaps.
            fill_centers = [
                (position.center_x, position.center_y)
                for position in missing_positions
            ][: expected_slots - len(balls)]
        else:
            for b in balls:
                col = min(range(len(grid_xs)), key=lambda i, bx=b.center_x: abs(grid_xs[i] - bx))
                row = min(range(len(grid_ys)), key=lambda j, by=b.center_y: abs(grid_ys[j] - by))
                occupied_slots.add((row, col))
        next_id = len(balls) + 1
        seed_r = max(8, int(round(final_pitch * config.roi_pad_radius_seed_pitch_ratio)))
        r_min = int(round(final_pitch * config.roi_pad_radius_min_pitch_ratio))
        r_max = int(round(final_pitch * config.roi_pad_radius_max_pitch_ratio))
        if fill_centers is not None:
            fill_targets = fill_centers
        else:
            fill_targets = [
                (int(round(gx)), int(round(gy)))
                for row_idx, gy in enumerate(grid_ys)
                for col_idx, gx in enumerate(grid_xs)
                if (row_idx, col_idx) not in occupied_slots
            ]
        for cx, cy in fill_targets:
            r = _estimate_pad_radius(image, cx, cy, seed_r, config)
            r = max(r_min, min(r, r_max))
            ball_is_estimated = True
            if config.use_local_component_final_snap:
                snapped_cx, snapped_cy, snapped_r = _snap_pad_center_to_local_component(
                    image=image,
                    x=cx,
                    y=cy,
                    radius=r,
                    pitch=final_pitch,
                    config=config,
                )
                if snapped_cx != cx or snapped_cy != cy:
                    cx, cy, r = snapped_cx, snapped_cy, snapped_r
                    ball_is_estimated = False
            balls.append(
                SolderBall(
                    ball_id=next_id,
                    center_x=cx,
                    center_y=cy,
                    radius=r,
                    confidence=0.20 if ball_is_estimated else 0.65,
                    is_estimated=ball_is_estimated,
                )
            )
            next_id += 1

    # Post-assembly per-pad refinement: refine ALL ball positions (real + estimated)
    # using local dark-blob search WITHOUT re-running dedup. Fixes slightly off-center
    # circles without causing detection merging.
    # Uses ROI-masked image so post_refine never snaps to structures outside the BGA square.
    apply_post_snap = config.use_post_refine_local_snap or (
        config.auto_post_snap_anisotropic_grids and grid_is_anisotropic
    )
    if apply_post_snap and final_pitch is not None:
        if roi is not None:
            refine_image = image.copy()
            h_img, w_img = image.shape[:2]
            if roi.y_min > 0:
                refine_image[: roi.y_min, :] = 128
            if roi.y_max < h_img:
                refine_image[roi.y_max :, :] = 128
            if roi.x_min > 0:
                refine_image[:, : roi.x_min] = 128
            if roi.x_max < w_img:
                refine_image[:, roi.x_max :] = 128
        else:
            refine_image = image
        balls = _post_refine_ball_positions(refine_image, balls, final_pitch, config)

    final_circles = {candidate.circle for candidate in final_candidates}
    rejected_candidates = _rejected_real_candidates(
        candidates=raw_candidates,
        kept_candidates=final_circles,
        roi=roi,
    )
    diagnostics = BallDetectionDiagnostics(
        raw_candidates=tuple(candidate.circle for candidate in raw_candidates),
        bga_roi=roi,
        filtered_grid=tuple((ball.center_x, ball.center_y, ball.radius) for ball in balls),
        rejected_candidates=tuple(rejected_candidates),
        missing_grid_positions=tuple(missing_positions),
        occupied_grid_slots=len(balls),
        expected_grid_slots=expected_slots,
    )
    return BallDetectionResult(balls=balls, diagnostics=diagnostics)


def _post_refine_ball_positions(
    image: np.ndarray,
    balls: list[SolderBall],
    pitch: float,
    config: HoughCircleConfig,
) -> list[SolderBall]:
    """Refine every ball center to the nearest local dark blob — no dedup, no count change.

    Applied after full grid assembly so that all 256 positions are independently
    centered on their actual pad without risking duplicate-merging from dedup.
    Balls that find a local blob become real; others keep their current status.
    """
    from dataclasses import replace as dc_replace

    min_separation = pitch * 0.45
    refined: list[SolderBall] = []
    for index, ball in enumerate(balls):
        snapped_cx, snapped_cy, snapped_r = _snap_pad_center_to_local_component(
            image=image,
            x=ball.center_x,
            y=ball.center_y,
            radius=ball.radius,
            pitch=pitch,
            config=config,
        )
        if snapped_cx != ball.center_x or snapped_cy != ball.center_y:
            # Reject snaps that converge onto another ball's position:
            # each grid slot must keep exactly one detection.
            others = [
                (b.center_x, b.center_y)
                for b in (*refined, *balls[index + 1 :])
            ]
            if any(
                np.hypot(snapped_cx - ox, snapped_cy - oy) < min_separation
                for ox, oy in others
            ):
                refined.append(ball)
                continue
        if snapped_cx != ball.center_x or snapped_cy != ball.center_y:
            refined.append(
                dc_replace(
                    ball,
                    center_x=snapped_cx,
                    center_y=snapped_cy,
                    radius=snapped_r,
                    is_estimated=False,
                )
            )
        else:
            refined.append(ball)
    return refined


def _detect_hough_solder_balls(
    image: np.ndarray,
    config: HoughCircleConfig,
) -> list[SolderBall]:
    """Detect dark circular BGA solder balls with the previous Hough flow."""
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=config.dp,
        minDist=config.min_dist,
        param1=config.param1,
        param2=config.param2,
        minRadius=config.min_radius,
        maxRadius=config.max_radius,
    )
    if circles is None:
        return []

    candidates = np.round(circles[0]).astype(int)
    filtered = [
        (int(x), int(y), int(radius))
        for x, y, radius in candidates
        if _is_dark_circle(image, int(x), int(y), int(radius), config)
    ]
    filtered = _remove_duplicate_circles(filtered, config)
    layout_filtered = _remove_layout_outliers(filtered, config)
    if len(layout_filtered) >= len(filtered) * config.min_layout_keep_ratio:
        filtered = layout_filtered
    if config.recover_grid_gaps:
        filtered = _recover_grid_gap_circles(image, filtered, config)
    if config.regularize_grid_centers:
        filtered = _regularize_grid_centers(filtered, config)
    if config.normalize_small_pad_radii:
        filtered = _normalize_small_pad_radii(filtered, config)
    if config.refine_boundary:
        filtered = _refine_circle_boundaries(image, filtered, config)
    if config.normalize_contour_radii:
        filtered = _normalize_contour_radii(filtered, config)
    filtered.sort(key=lambda item: (item[1], item[0]))

    return [
        SolderBall(
            ball_id=index + 1,
            center_x=x,
            center_y=y,
            radius=radius,
        )
        for index, (x, y, radius) in enumerate(filtered)
    ]


def crop_ball_roi(
    image: np.ndarray,
    ball: SolderBall,
) -> tuple[np.ndarray, RoiBounds, tuple[int, int]]:
    """Crop a square ROI around one detected solder ball."""
    height, width = image.shape[:2]
    x_min = max(ball.center_x - ball.radius, 0)
    y_min = max(ball.center_y - ball.radius, 0)
    x_max = min(ball.center_x + ball.radius + 1, width)
    y_max = min(ball.center_y + ball.radius + 1, height)

    bounds = RoiBounds(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
    roi = image[y_min:y_max, x_min:x_max]
    local_center = (ball.center_x - x_min, ball.center_y - y_min)
    return roi, bounds, local_center


def _detect_blob_pad_candidates(
    image: np.ndarray,
    config: HoughCircleConfig,
) -> list[_PadCandidate]:
    """Detect dark, compact pad-like components for BGA ROI estimation."""
    height, width = image.shape[:2]
    scale = min(1.0, config.roi_blob_detection_max_dimension / max(height, width))
    if scale < 1.0:
        small = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image

    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[_PadCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < config.roi_blob_min_area or area > config.roi_blob_max_area:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)
        aspect_ratio = max(box_width, box_height) / max(1, min(box_width, box_height))
        if aspect_ratio > config.roi_blob_max_aspect_ratio:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        circularity = (4.0 * pi * area) / (perimeter**2) if perimeter else 0.0
        if circularity < config.roi_blob_min_circularity:
            continue

        extent = area / max(1.0, float(box_width * box_height))
        if extent < config.roi_blob_min_extent:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        center_x = int(round((moments["m10"] / moments["m00"]) / scale))
        center_y = int(round((moments["m01"] / moments["m00"]) / scale))
        radius = max(1, int(round((np.sqrt(area / pi)) / scale)))
        score = float(circularity + extent)
        candidates.append(
            _PadCandidate(
                x=center_x,
                y=center_y,
                radius=radius,
                score=score,
                source="blob",
            ),
        )

    return _merge_pad_candidates(candidates, config)


def _pad_candidates_from_balls(balls: list[SolderBall]) -> list[_PadCandidate]:
    """Convert Hough solder balls to real pad candidates."""
    return [
        _PadCandidate(
            x=ball.center_x,
            y=ball.center_y,
            radius=ball.radius,
            score=2.5,
            source="hough",
        )
        for ball in balls
    ]


def _merge_pad_candidates(
    candidates: list[_PadCandidate],
    config: HoughCircleConfig,
) -> list[_PadCandidate]:
    """Merge duplicate local detections while keeping the strongest evidence."""
    merged: list[_PadCandidate] = []
    ordered = sorted(
        candidates,
        key=lambda item: (item.score, item.radius),
        reverse=True,
    )
    for candidate in ordered:
        duplicate_index = None
        for index, existing in enumerate(merged):
            distance = float(np.hypot(candidate.x - existing.x, candidate.y - existing.y))
            threshold = (
                min(candidate.radius, existing.radius)
                * config.duplicate_distance_factor
            )
            if distance <= max(6.0, threshold):
                duplicate_index = index
                break

        if duplicate_index is None:
            merged.append(candidate)
            continue

        existing = merged[duplicate_index]
        if candidate.score > existing.score:
            merged[duplicate_index] = candidate

    return sorted(merged, key=lambda item: (item.y, item.x))


def _estimate_bga_roi_from_blob_candidates(
    candidates: list[_PadCandidate],
    image_shape: tuple[int, int],
    config: HoughCircleConfig,
) -> _BgaRoiEstimate | None:
    """Estimate the BGA package square from the densest regular pad component."""
    if len(candidates) < max(30, config.roi_blob_min_component_candidates // 2):
        return None

    radii = np.array([candidate.radius for candidate in candidates], dtype=np.float32)
    reference_radius = float(np.percentile(radii, 65))
    if reference_radius <= 0.0:
        return None

    radius_filtered = [
        candidate
        for candidate in candidates
        if reference_radius * config.roi_blob_radius_min_ratio
        <= candidate.radius
        <= reference_radius * config.roi_blob_radius_max_ratio
    ]
    if len(radius_filtered) < max(30, config.roi_blob_min_component_candidates // 2):
        return None

    points = np.array(
        [(candidate.x, candidate.y) for candidate in radius_filtered],
        dtype=np.float32,
    )
    best: _BgaRoiEstimate | None = None
    best_score = -np.inf
    for pitch in _blob_pitch_candidates(points, config):
        pitch_radius_ratio = pitch / max(1.0, reference_radius)
        if pitch_radius_ratio < config.roi_blob_pitch_min_radius_ratio:
            continue
        if pitch_radius_ratio > config.roi_blob_pitch_max_radius_ratio:
            continue

        component_indexes = _largest_grid_neighbor_component(points, pitch, config)
        if len(component_indexes) < config.roi_blob_min_component_candidates:
            continue

        component_points = points[component_indexes]
        quantile = config.roi_blob_quantile
        x0, y0 = np.quantile(component_points, quantile, axis=0)
        x1, y1 = np.quantile(component_points, 1.0 - quantile, axis=0)
        observed_width = float(x1 - x0)
        observed_height = float(y1 - y0)
        grid_width = max(observed_width, (config.expected_grid_cols - 1) * pitch)
        grid_height = max(observed_height, (config.expected_grid_rows - 1) * pitch)
        center_x = float((x0 + x1) / 2.0)
        center_y = float((y0 + y1) / 2.0)
        margin = pitch * config.roi_blob_margin_pitch_ratio
        roi = _clamp_roi(
            RoiBounds(
                x_min=int(round(center_x - (grid_width / 2.0) - margin)),
                y_min=int(round(center_y - (grid_height / 2.0) - margin)),
                x_max=int(round(center_x + (grid_width / 2.0) + margin)),
                y_max=int(round(center_y + (grid_height / 2.0) + margin)),
            ),
            image_shape,
        )
        area = float((roi.x_max - roi.x_min) * (roi.y_max - roi.y_min))
        fill = len(component_indexes) / max(1.0, area / (pitch * pitch))
        pitch_penalty = pitch_radius_ratio * 25.0
        score = (len(component_indexes) * 100.0) + (fill * 1000.0)
        score -= (area * 0.002) + pitch_penalty
        if score > best_score:
            best_score = score
            best = _BgaRoiEstimate(
                roi=roi,
                pitch=float(pitch),
                support_count=len(component_indexes),
                source="blob-grid",
            )

    return best


def _blob_pitch_candidates(
    points: np.ndarray,
    config: HoughCircleConfig,
) -> list[float]:
    """Estimate pitch candidates from nearly aligned neighboring blobs."""
    if len(points) < 2:
        return []

    sampled_points = points
    if len(sampled_points) > 900:
        indexes = np.linspace(0, len(sampled_points) - 1, 900).astype(int)
        sampled_points = sampled_points[indexes]

    gaps: list[float] = []
    min_pitch = max(35.0, config.grid_search_min_pitch * 0.75)
    max_pitch = config.grid_search_max_pitch
    for tolerance in (10.0, 14.0, 18.0, 24.0, 30.0):
        for x, y in sampled_points:
            dx = np.abs(sampled_points[:, 0] - x)
            dy = np.abs(sampled_points[:, 1] - y)
            row_gaps = dx[
                (dy <= tolerance)
                & (dx >= min_pitch)
                & (dx <= max_pitch)
            ]
            column_gaps = dy[
                (dx <= tolerance)
                & (dy >= min_pitch)
                & (dy <= max_pitch)
            ]
            gaps.extend(row_gaps.tolist())
            gaps.extend(column_gaps.tolist())

    if not gaps:
        return []

    bin_width = config.grid_pitch_histogram_bin
    bins = np.arange(min_pitch, max_pitch + bin_width, bin_width)
    histogram, edges = np.histogram(gaps, bins=bins)
    pitches: list[float] = []
    for index in np.argsort(histogram)[-8:][::-1]:
        if histogram[index] < 4:
            continue

        pitch = float((edges[index] + edges[index + 1]) / 2.0)
        if all(abs(pitch - existing) > bin_width * 2.5 for existing in pitches):
            pitches.append(pitch)

    return pitches


def _largest_grid_neighbor_component(
    points: np.ndarray,
    pitch: float,
    config: HoughCircleConfig,
) -> list[int]:
    """Return the largest component linked by row/column pitch neighbors."""
    row_tolerance = max(12.0, pitch * config.roi_neighbor_tolerance_ratio)
    min_gap = pitch * config.roi_neighbor_min_pitch_ratio
    max_gap = pitch * config.roi_neighbor_max_pitch_ratio
    edges: dict[int, list[int]] = defaultdict(list)
    for index, (x, y) in enumerate(points):
        dx = np.abs(points[:, 0] - x)
        dy = np.abs(points[:, 1] - y)
        near = (
            (dy <= row_tolerance) & (dx >= min_gap) & (dx <= max_gap)
        ) | (
            (dx <= row_tolerance) & (dy >= min_gap) & (dy <= max_gap)
        )
        for neighbor in np.where(near)[0]:
            if int(neighbor) != index:
                edges[index].append(int(neighbor))

    seen: set[int] = set()
    components: list[list[int]] = []
    for index in range(len(points)):
        if index in seen:
            continue

        queue: deque[int] = deque([index])
        seen.add(index)
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in edges[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    components.sort(key=len, reverse=True)
    return components[0] if components else []


def _detect_roi_hough_pad_candidates(
    image: np.ndarray,
    roi_estimate: _BgaRoiEstimate,
    config: HoughCircleConfig,
) -> list[_PadCandidate]:
    """Run a local Hough search inside the estimated BGA ROI."""
    roi = roi_estimate.roi
    pitch = roi_estimate.pitch
    crop = image[roi.y_min : roi.y_max, roi.x_min : roi.x_max]
    if crop.size == 0:
        return []

    seed_radius = max(1.0, pitch * config.roi_pad_radius_seed_pitch_ratio)
    zoom = max(1.0, min(2.2, 38.0 / seed_radius))
    if zoom > 1.01:
        search_image = cv2.resize(
            crop,
            None,
            fx=zoom,
            fy=zoom,
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        search_image = crop

    min_radius = max(5, int(round(pitch * 0.16 * zoom)))
    max_radius = max(min_radius + 2, int(round(pitch * 0.48 * zoom)))
    min_distance = max(10.0, pitch * 0.55 * zoom)
    circles = cv2.HoughCircles(
        search_image,
        cv2.HOUGH_GRADIENT,
        dp=config.dp,
        minDist=min_distance,
        param1=config.param1,
        param2=18.0,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []

    candidates: list[_PadCandidate] = []
    for x, y, radius in np.round(circles[0]).astype(int):
        center_x = int(round((x / zoom) + roi.x_min))
        center_y = int(round((y / zoom) + roi.y_min))
        full_radius = max(1, int(round(radius / zoom)))
        if not _roi_contains_point(roi, center_x, center_y):
            continue

        evidence_radius = max(
            full_radius,
            int(round(pitch * config.roi_pad_radius_seed_pitch_ratio)),
        )
        evidence_score = _grid_slot_evidence_score(
            image=image,
            x=center_x,
            y=center_y,
            radius=evidence_radius,
        )
        if evidence_score < 0.25:
            continue
        metrics = _dark_component_geometry_metrics(
            image=image,
            x=center_x,
            y=center_y,
            radius=evidence_radius,
            config=config,
            intensity_threshold=150,
        )
        if metrics is None:
            continue

        aspect_ratio, area_ratio, circularity = metrics
        if area_ratio < 0.18:
            continue
        if aspect_ratio > 2.10 and circularity < 0.35:
            continue

        candidates.append(
            _PadCandidate(
                x=center_x,
                y=center_y,
                radius=full_radius,
                score=2.0 + evidence_score,
                source="roi-hough",
            ),
        )

    return _merge_pad_candidates(candidates, config)


def _large_roi_candidates(
    candidates: list[_PadCandidate],
    roi: RoiBounds,
) -> list[_PadCandidate]:
    """Return likely solder-pad candidates, ignoring small via-like blobs."""
    inside = [
        candidate
        for candidate in candidates
        if _roi_contains_point(roi, candidate.x, candidate.y)
    ]
    if not inside:
        return []

    radii = np.array([candidate.radius for candidate in inside], dtype=np.float32)
    reference_radius = float(np.percentile(radii, 65))
    radius_floor = max(1.0, reference_radius * 0.62)
    return [candidate for candidate in inside if candidate.radius >= radius_floor]


def _detect_roi_component_pad_candidates(
    image: np.ndarray,
    roi: RoiBounds,
    pitch: float,
    config: HoughCircleConfig,
) -> list[_PadCandidate]:
    """Detect elongated dark pads directly as ROI-local components."""
    crop = image[roi.y_min : roi.y_max, roi.x_min : roi.x_max]
    if crop.size == 0 or pitch <= 0.0:
        return []

    blurred = cv2.GaussianBlur(crop, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    min_area = max(120.0, (pitch**2) * config.roi_component_pad_min_area_ratio)
    max_area = max(min_area + 1.0, (pitch**2) * config.roi_component_pad_max_area_ratio)
    candidates: list[_PadCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)
        aspect_ratio = max(box_width, box_height) / max(1, min(box_width, box_height))
        if aspect_ratio > config.roi_component_pad_max_aspect_ratio:
            continue

        extent = area / max(1.0, float(box_width * box_height))
        if extent < config.roi_component_pad_min_extent:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        circularity = (4.0 * pi * area) / (perimeter**2) if perimeter else 0.0
        if circularity < config.roi_component_pad_min_circularity:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        center_x = int(round(moments["m10"] / moments["m00"])) + roi.x_min
        center_y = int(round(moments["m01"] / moments["m00"])) + roi.y_min
        radius = max(1, int(round(np.sqrt(area / pi))))
        candidates.append(
            _PadCandidate(
                x=center_x,
                y=center_y,
                radius=radius,
                score=1.8 + float(circularity + extent),
                source="roi-component",
            ),
        )

    return _merge_pad_candidates(candidates, config)


def _should_prefer_roi_component_candidates(
    current_candidates: list[_PadCandidate],
    component_candidates: list[_PadCandidate],
    expected_slots: int,
    config: HoughCircleConfig,
) -> bool:
    """Prefer local components when Hough is incomplete and visibly unstable."""
    if not component_candidates:
        return False

    min_component_count = int(
        round(expected_slots * config.roi_component_candidate_min_result_ratio),
    )
    if len(component_candidates) < min_component_count:
        return False

    replacement_floor = int(
        round(len(current_candidates) * config.roi_component_candidate_replace_ratio),
    )
    return len(component_candidates) >= replacement_floor


def _regularize_component_candidates_on_axis_grid(
    image: np.ndarray,
    candidates: list[_PadCandidate],
    pitch: float,
    config: HoughCircleConfig,
) -> tuple[list[_PadCandidate], list[MissingGridPosition]]:
    """Keep one component per 16x16 axis-grid slot and recover strong gaps."""
    grid = _fit_axis_cluster_grid_from_candidates(
        candidates=candidates,
        pitch=pitch,
        config=config,
    )
    if grid is None:
        return candidates, []

    assignments = _assign_candidates_to_axis_grid(
        [candidate.circle for candidate in candidates],
        grid,
        config,
        tolerance_ratio=config.roi_component_axis_assign_tolerance_ratio,
    )
    used_indexes = set(assignments.values())
    assigned_candidates = [
        candidate
        for index, candidate in enumerate(candidates)
        if index in used_indexes
    ]
    recovered_candidates = _recover_grid_evidence_candidates(
        image=image,
        grid=grid,
        assignments=assignments,
        config=config,
        min_score=config.roi_component_grid_evidence_min_score,
    )
    expected_slots = config.expected_grid_rows * config.expected_grid_cols
    max_recovered = int(round(expected_slots * config.grid_evidence_max_recovered_ratio))
    if len(recovered_candidates) > max_recovered:
        recovered_candidates = []

    regularized = _merge_pad_candidates(
        [*assigned_candidates, *recovered_candidates],
        config,
    )
    final_assignments = _assign_candidates_to_axis_grid(
        [candidate.circle for candidate in regularized],
        grid,
        config,
        tolerance_ratio=config.roi_component_axis_assign_tolerance_ratio,
    )
    missing_positions = _missing_positions_from_grid(
        grid=grid,
        assignments=final_assignments,
        config=config,
    )
    return regularized, missing_positions


def _regularize_final_candidates_on_roi_grid(
    image: np.ndarray,
    candidates: list[_PadCandidate],
    roi: RoiBounds,
    pitch: float,
    config: HoughCircleConfig,
) -> tuple[list[_PadCandidate], list[MissingGridPosition], float, _BgaGridFit | None]:
    """Constrain final pad detections to one coherent 16x16 grid inside the ROI."""
    expected_slots = config.expected_grid_rows * config.expected_grid_cols
    component_candidates = _detect_roi_component_pad_candidates(
        image=image,
        roi=roi,
        pitch=pitch,
        config=config,
    )
    pool = _merge_pad_candidates(
        [
            *candidates,
            *component_candidates,
        ],
        config,
    )
    full_support_pool = _roi_grid_support_candidates(
        image=image,
        candidates=pool,
        roi=roi,
        pitch=pitch,
        config=config,
    )
    component_support_pool = _roi_grid_support_candidates(
        image=image,
        candidates=component_candidates,
        roi=roi,
        pitch=pitch,
        config=config,
    )
    if not full_support_pool and not component_support_pool:
        return [], [], pitch, None

    fit_pool = full_support_pool
    if len(component_support_pool) >= int(
        round(expected_slots * config.final_grid_component_fit_min_ratio),
    ):
        fit_pool = component_support_pool

    grid = _fit_final_roi_axis_grid(
        image=image,
        candidates=fit_pool,
        roi=roi,
        pitch=pitch,
        config=config,
    )
    if grid is None:
        return full_support_pool or component_support_pool, [], pitch, None

    assignment_pool = full_support_pool
    if len(component_support_pool) >= int(
        round(expected_slots * config.final_grid_component_assign_min_ratio),
    ):
        assignment_pool = component_support_pool
    assignments = _assign_pad_candidates_to_axis_grid(
        image=image,
        candidates=assignment_pool,
        grid=grid,
        config=config,
        tolerance_ratio=config.final_grid_assign_tolerance_ratio,
    )
    min_assignments = int(round(expected_slots * config.final_grid_min_assignment_ratio))
    if len(assignments) < min_assignments:
        return assignment_pool, [], pitch, None

    use_affine = config.use_affine_final_grid
    if (
        not use_affine
        and config.auto_affine_anisotropic_grids
        and grid.pitch_x > 0
        and grid.pitch_y > 0
    ):
        # Oblique views shear the grid (columns lean with row), which an
        # axis-aligned grid cannot follow; the affine model can.
        grid_anisotropy = max(grid.pitch_x, grid.pitch_y) / min(
            grid.pitch_x,
            grid.pitch_y,
        )
        use_affine = grid_anisotropy >= config.post_snap_min_pitch_anisotropy
    if use_affine:
        affine_result = _regularize_candidates_on_affine_grid(
            image=image,
            candidates=assignment_pool,
            roi=roi,
            seed_grid=grid,
            seed_assignments=assignments,
            config=config,
        )
        if affine_result is not None:
            affine_candidates, affine_missing, affine_pitch = affine_result
            if len(affine_candidates) >= int(round(expected_slots * 0.90)):
                return (
                    sorted(
                        affine_candidates,
                        key=lambda candidate: (candidate.y, candidate.x),
                    ),
                    affine_missing,
                    affine_pitch,
                    grid,
                )

    regularized = _pad_candidates_from_axis_grid_assignments(
        image=image,
        candidates=assignment_pool,
        grid=grid,
        assignments=assignments,
        config=config,
    )
    if config.final_grid_recover_candidates:
        recovered_candidates = _recover_slot_local_pad_candidates(
            image=image,
            roi=roi,
            grid=grid,
            assignments=assignments,
            config=config,
            assigned_candidates=regularized,
        )
        max_recovered = int(round(expected_slots * config.grid_evidence_max_recovered_ratio))
        if 0 < len(recovered_candidates) <= max_recovered:
            combined_candidates = [*regularized, *recovered_candidates]
            combined_assignments = _assign_pad_candidates_to_axis_grid(
                image=image,
                candidates=combined_candidates,
                grid=grid,
                config=config,
                tolerance_ratio=config.final_grid_assign_tolerance_ratio,
            )
            used_indexes = set(combined_assignments.values())
            regularized = [
                candidate
                for index, candidate in enumerate(combined_candidates)
                if index in used_indexes
            ]

    final_assignments = _assign_pad_candidates_to_axis_grid(
        image=image,
        candidates=regularized,
        grid=grid,
        config=config,
        tolerance_ratio=config.final_grid_assign_tolerance_ratio,
    )
    missing_positions = _missing_positions_from_grid(
        grid=grid,
        assignments=final_assignments,
        config=config,
    )
    return (
        sorted(regularized, key=lambda candidate: (candidate.y, candidate.x)),
        missing_positions,
        float((grid.pitch_x + grid.pitch_y) / 2.0),
        grid,
    )


def _regularize_candidates_on_affine_grid(
    image: np.ndarray,
    candidates: list[_PadCandidate],
    roi: RoiBounds,
    seed_grid: _BgaGridFit,
    seed_assignments: dict[tuple[int, int], int],
    config: HoughCircleConfig,
) -> tuple[list[_PadCandidate], list[MissingGridPosition], float] | None:
    """Fit a mildly skewed 16x16 grid and keep one real candidate per slot."""
    expected_slots = config.expected_grid_rows * config.expected_grid_cols
    if len(seed_assignments) < int(round(expected_slots * 0.70)):
        return None

    pitch = float((seed_grid.pitch_x + seed_grid.pitch_y) / 2.0)
    src_points: list[tuple[float, float]] = []
    dst_points: list[tuple[float, float]] = []
    for (row, column), index in seed_assignments.items():
        candidate = candidates[index]
        seed_radius = max(
            candidate.radius,
            int(round(pitch * config.roi_pad_radius_seed_pitch_ratio)),
        )
        evidence_score = _grid_slot_evidence_score(
            image=image,
            x=candidate.x,
            y=candidate.y,
            radius=seed_radius,
        )
        if evidence_score < 0.26:
            continue

        src_points.append((float(column), float(row)))
        dst_points.append((float(candidate.x), float(candidate.y)))

    if len(src_points) < int(round(expected_slots * 0.55)):
        return None

    source_array = np.array(src_points, dtype=np.float32)
    destination_array = np.array(dst_points, dtype=np.float32)
    affine, inliers = cv2.estimateAffine2D(
        source_array,
        destination_array,
        method=cv2.RANSAC,
        ransacReprojThreshold=max(8.0, pitch * 0.28),
        maxIters=3000,
        confidence=0.995,
        refineIters=12,
    )
    if affine is None or inliers is None:
        return None

    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < int(round(expected_slots * 0.45)):
        return None

    expected_centers = _affine_grid_centers(affine, roi, config)
    if len(expected_centers) != expected_slots:
        return None

    vector_x = affine[:, 0]
    vector_y = affine[:, 1]
    affine_pitch = float((np.linalg.norm(vector_x) + np.linalg.norm(vector_y)) / 2.0)
    if affine_pitch <= 0.0:
        return None

    assigned_candidates, assigned_slots = _assign_candidates_to_affine_slots(
        image=image,
        candidates=candidates,
        expected_centers=expected_centers,
        roi=roi,
        pitch=affine_pitch,
        config=config,
    )
    if len(assigned_candidates) < int(round(expected_slots * 0.70)):
        return None

    refined_centers = _refine_affine_centers_with_homography(
        assigned_candidates=assigned_candidates,
        expected_centers=expected_centers,
        assigned_slots=assigned_slots,
        roi=roi,
        pitch=affine_pitch,
        config=config,
    )
    if refined_centers is not None:
        refined_assigned, refined_slots = _assign_candidates_to_affine_slots(
            image=image,
            candidates=candidates,
            expected_centers=refined_centers,
            roi=roi,
            pitch=affine_pitch,
            config=config,
        )
        if len(refined_assigned) >= len(assigned_candidates):
            expected_centers = refined_centers
            assigned_candidates = refined_assigned
            assigned_slots = refined_slots

    recovered_candidates = _recover_affine_slot_local_pad_candidates(
        image=image,
        roi=roi,
        expected_centers=expected_centers,
        assigned_slots=assigned_slots,
        assigned_candidates=assigned_candidates,
        pitch=affine_pitch,
        config=config,
    )
    if recovered_candidates:
        combined = [*assigned_candidates, *recovered_candidates]
        assigned_candidates, assigned_slots = _assign_candidates_to_affine_slots(
            image=image,
            candidates=combined,
            expected_centers=expected_centers,
            roi=roi,
            pitch=affine_pitch,
            config=config,
            allow_grid_evidence=True,
        )

    merged_candidates = _merge_pad_candidates(assigned_candidates, config)

    # Cell-level dedup with slot accounting: sheared assignments can leave
    # two candidates inside one grid cell (ball + via shadow). Keep the
    # stronger one and release the losing slot so it is reported (and later
    # filled) as missing instead of silently dropping a pad from the grid.
    threshold_x = float(np.linalg.norm(vector_x)) * 0.55
    threshold_y = float(np.linalg.norm(vector_y)) * 0.55
    deduped_candidates: list[_PadCandidate] = []
    dropped_candidates: list[_PadCandidate] = []
    for candidate in sorted(
        merged_candidates,
        key=lambda item: item.score,
        reverse=True,
    ):
        clash = any(
            abs(candidate.x - kept.x) < threshold_x
            and abs(candidate.y - kept.y) < threshold_y
            for kept in deduped_candidates
        )
        if clash:
            dropped_candidates.append(candidate)
        else:
            deduped_candidates.append(candidate)
    dropped_candidates.extend(
        candidate
        for candidate in assigned_candidates
        if (candidate.x, candidate.y)
        not in {(kept.x, kept.y) for kept in merged_candidates}
    )
    if dropped_candidates:
        slot_items = [
            (slot, center)
            for slot, center in expected_centers.items()
            if slot in assigned_slots
        ]
        for candidate in dropped_candidates:
            dropped_slot, _center = min(
                slot_items,
                key=lambda item: (candidate.x - item[1][0]) ** 2
                + (candidate.y - item[1][1]) ** 2,
            )
            assigned_slots.discard(dropped_slot)
    merged_candidates = sorted(
        deduped_candidates,
        key=lambda item: (item.y, item.x),
    )
    missing_positions = _missing_positions_from_affine_slots(
        expected_centers=expected_centers,
        assigned_slots=assigned_slots,
        config=config,
    )
    return (
        merged_candidates,
        missing_positions,
        affine_pitch,
    )


def _affine_grid_centers(
    affine: np.ndarray,
    roi: RoiBounds,
    config: HoughCircleConfig,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Project every 16x16 grid slot through an affine transform."""
    centers: dict[tuple[int, int], tuple[int, int]] = {}
    for row in range(config.expected_grid_rows):
        for column in range(config.expected_grid_cols):
            source = np.array([float(column), float(row), 1.0], dtype=np.float32)
            center_x, center_y = affine @ source
            x = int(round(float(center_x)))
            y = int(round(float(center_y)))
            if not _roi_contains_point(roi, x, y):
                continue
            centers[(row, column)] = (x, y)
    return centers


def _refine_affine_centers_with_homography(
    assigned_candidates: list[_PadCandidate],
    expected_centers: dict[tuple[int, int], tuple[int, int]],
    assigned_slots: set[tuple[int, int]],
    roi: RoiBounds,
    pitch: float,
    config: HoughCircleConfig,
) -> dict[tuple[int, int], tuple[int, int]] | None:
    """Re-project the 16x16 slots through a perspective (homography) fit.

    The affine fit absorbs the linear part of an oblique view, but true
    perspective bends the grid non-linearly, so the residual error peaks in
    the corners. A homography is the exact model for a planar grid under
    perspective, so slots re-projected through it land on the corner pads.
    """
    expected_slots = config.expected_grid_rows * config.expected_grid_cols
    slot_centers = [
        (slot, center)
        for slot, center in expected_centers.items()
        if slot in assigned_slots
    ]
    if not slot_centers:
        return None

    max_pair_distance = pitch * 0.60
    src_points: list[tuple[float, float]] = []
    dst_points: list[tuple[float, float]] = []
    for candidate in assigned_candidates:
        (row, column), (center_x, center_y) = min(
            slot_centers,
            key=lambda item: (candidate.x - item[1][0]) ** 2
            + (candidate.y - item[1][1]) ** 2,
        )
        if np.hypot(candidate.x - center_x, candidate.y - center_y) > max_pair_distance:
            continue
        src_points.append((float(column), float(row)))
        dst_points.append((float(candidate.x), float(candidate.y)))

    if len(src_points) < int(round(expected_slots * 0.55)):
        return None

    homography, inlier_mask = cv2.findHomography(
        np.array(src_points, dtype=np.float32),
        np.array(dst_points, dtype=np.float32),
        cv2.RANSAC,
        max(4.0, pitch * 0.15),
        maxIters=5000,
        confidence=0.998,
    )
    if homography is None or inlier_mask is None:
        return None
    if int(np.count_nonzero(inlier_mask)) < int(round(len(src_points) * 0.80)):
        return None

    centers: dict[tuple[int, int], tuple[int, int]] = {}
    for row in range(config.expected_grid_rows):
        for column in range(config.expected_grid_cols):
            source = np.array([float(column), float(row), 1.0], dtype=np.float64)
            projected = homography @ source
            if abs(projected[2]) < 1e-9:
                return None
            x = int(round(float(projected[0] / projected[2])))
            y = int(round(float(projected[1] / projected[2])))
            if not _roi_contains_point(roi, x, y):
                continue
            centers[(row, column)] = (x, y)

    if len(centers) != expected_slots:
        return None
    return centers


def _assign_candidates_to_affine_slots(
    image: np.ndarray,
    candidates: list[_PadCandidate],
    expected_centers: dict[tuple[int, int], tuple[int, int]],
    roi: RoiBounds,
    pitch: float,
    config: HoughCircleConfig,
    allow_grid_evidence: bool = False,
) -> tuple[list[_PadCandidate], set[tuple[int, int]]]:
    """Greedily assign local evidence candidates to affine grid positions."""
    if not candidates or not expected_centers:
        return [], set()

    tolerance = max(10.0, pitch * 0.46)
    seed_radius = max(
        config.grid_radius_min,
        int(round(pitch * config.roi_pad_radius_seed_pitch_ratio)),
    )
    min_radius = int(round(pitch * config.roi_pad_radius_min_pitch_ratio))
    max_radius = int(round(pitch * config.roi_pad_radius_max_pitch_ratio))
    pair_scores: list[tuple[float, tuple[int, int], int, float]] = []
    for slot, (expected_x, expected_y) in expected_centers.items():
        for index, candidate in enumerate(candidates):
            if candidate.source == "grid-evidence" and not allow_grid_evidence:
                continue
            if not _roi_contains_point(roi, candidate.x, candidate.y):
                continue
            if not _roi_contains_circle_center_band(
                roi=roi,
                x=candidate.x,
                y=candidate.y,
                radius=max(min_radius, min(candidate.radius, max_radius)),
            ):
                continue

            distance = float(
                np.hypot(candidate.x - expected_x, candidate.y - expected_y),
            )
            if distance > tolerance:
                continue

            evidence_score = _grid_slot_evidence_score(
                image=image,
                x=candidate.x,
                y=candidate.y,
                radius=max(seed_radius, candidate.radius),
            )
            if evidence_score < config.final_grid_candidate_min_score:
                continue

            metrics = _dark_component_geometry_metrics(
                image=image,
                x=candidate.x,
                y=candidate.y,
                radius=max(seed_radius, candidate.radius),
                config=config,
                intensity_threshold=158,
            )
            if metrics is None:
                continue
            aspect_ratio, area_ratio, circularity = metrics
            if area_ratio < 0.08:
                continue
            if aspect_ratio > 3.85:
                continue
            if aspect_ratio > 2.55 and circularity < 0.08:
                continue

            distance_score = 1.0 - min(1.0, distance / tolerance)
            source_bonus = 0.16 if candidate.source in {"hough", "roi-hough"} else 0.0
            score = (
                (evidence_score * 2.0)
                + (distance_score * 0.8)
                + min(candidate.score, 4.0) * 0.12
                + source_bonus
            )
            pair_scores.append((score, slot, index, evidence_score))

    pair_scores.sort(key=lambda item: item[0], reverse=True)
    used_slots: set[tuple[int, int]] = set()
    used_candidates: set[int] = set()
    assigned: list[_PadCandidate] = []
    for score, slot, candidate_index, evidence_score in pair_scores:
        if slot in used_slots or candidate_index in used_candidates:
            continue

        candidate = candidates[candidate_index]
        radius = max(min_radius, min(candidate.radius, max_radius))
        assigned.append(
            _PadCandidate(
                x=candidate.x,
                y=candidate.y,
                radius=radius,
                score=candidate.score + evidence_score + score,
                source=candidate.source,
            ),
        )
        used_slots.add(slot)
        used_candidates.add(candidate_index)

    return assigned, used_slots


def _recover_affine_slot_local_pad_candidates(
    image: np.ndarray,
    roi: RoiBounds,
    expected_centers: dict[tuple[int, int], tuple[int, int]],
    assigned_slots: set[tuple[int, int]],
    assigned_candidates: list[_PadCandidate],
    pitch: float,
    config: HoughCircleConfig,
) -> list[_PadCandidate]:
    """Recover missing affine-grid slots from zoomed local pad evidence."""
    seed_radius = max(
        config.grid_radius_min,
        int(round(pitch * config.roi_pad_radius_seed_pitch_ratio)),
    )
    min_radius = int(round(pitch * config.roi_pad_radius_min_pitch_ratio))
    max_radius = int(round(pitch * config.roi_pad_radius_max_pitch_ratio))
    search_radius = max(
        seed_radius + 4,
        int(round(pitch * config.final_grid_slot_recovery_search_pitch_ratio)),
    )
    max_offset = max(
        8.0,
        pitch * config.final_grid_slot_recovery_max_offset_ratio,
    )
    zoom = max(
        1.0,
        min(
            config.final_grid_slot_recovery_max_zoom,
            config.final_grid_slot_recovery_zoom_target_radius / max(1.0, seed_radius),
        ),
    )
    height, width = image.shape[:2]
    assigned_points = np.array(
        [(candidate.x, candidate.y) for candidate in assigned_candidates],
        dtype=np.float32,
    )
    recovered: list[_PadCandidate] = []
    for slot, (expected_x, expected_y) in expected_centers.items():
        if slot in assigned_slots:
            continue
        if not _roi_contains_circle_center_band(
            roi=roi,
            x=expected_x,
            y=expected_y,
            radius=min_radius,
        ):
            continue

        if assigned_points.size:
            assigned_distances = np.hypot(
                assigned_points[:, 0] - expected_x,
                assigned_points[:, 1] - expected_y,
            )
            if float(np.min(assigned_distances)) < pitch * 0.18:
                continue

        x0 = max(roi.x_min, expected_x - search_radius)
        y0 = max(roi.y_min, expected_y - search_radius)
        x1 = min(roi.x_max, expected_x + search_radius)
        y1 = min(roi.y_max, expected_y + search_radius)
        if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
            continue
        if x1 <= x0 or y1 <= y0:
            continue

        local_crop = image[y0 : y1 + 1, x0 : x1 + 1]
        if zoom > 1.01:
            local_crop = cv2.resize(
                local_crop,
                None,
                fx=zoom,
                fy=zoom,
                interpolation=cv2.INTER_CUBIC,
            )

        candidate = _best_zoomed_slot_component_candidate(
            image=image,
            crop=local_crop,
            crop_origin=(x0, y0),
            zoom=zoom,
            expected_center=(expected_x, expected_y),
            seed_radius=seed_radius,
            pitch=pitch,
            min_radius=min_radius,
            max_radius=max_radius,
            max_offset=max_offset,
            config=config,
        )
        if candidate is None:
            candidate = _best_zoomed_slot_hough_candidate(
                image=image,
                crop=local_crop,
                crop_origin=(x0, y0),
                zoom=zoom,
                expected_center=(expected_x, expected_y),
                seed_radius=seed_radius,
                min_radius=min_radius,
                max_radius=max_radius,
                max_offset=max_offset,
                config=config,
            )
        if candidate is not None:
            recovered.append(candidate)

    return _merge_pad_candidates(recovered, config)


def _missing_positions_from_affine_slots(
    expected_centers: dict[tuple[int, int], tuple[int, int]],
    assigned_slots: set[tuple[int, int]],
    config: HoughCircleConfig,
) -> list[MissingGridPosition]:
    """Report unconfirmed affine-grid slots."""
    missing_positions: list[MissingGridPosition] = []
    for row in range(config.expected_grid_rows):
        for column in range(config.expected_grid_cols):
            slot = (row, column)
            if slot in assigned_slots:
                continue
            if slot not in expected_centers:
                continue
            x, y = expected_centers[slot]
            missing_positions.append(
                MissingGridPosition(
                    row=row + 1,
                    column=column + 1,
                    center_x=x,
                    center_y=y,
                    reason="expected affine 16x16 slot without direct pad candidate",
                ),
            )
    return missing_positions


def _roi_grid_support_candidates(
    image: np.ndarray,
    candidates: list[_PadCandidate],
    roi: RoiBounds,
    pitch: float,
    config: HoughCircleConfig,
) -> list[_PadCandidate]:
    """Keep only ROI-local candidates with pad evidence or pitch neighbors."""
    inside = [
        candidate
        for candidate in candidates
        if _roi_contains_point(roi, candidate.x, candidate.y)
        and candidate.source != "grid-evidence"
    ]
    if not inside:
        return []

    seed_radius = max(
        config.grid_radius_min,
        int(round(pitch * config.roi_pad_radius_seed_pitch_ratio)),
    )
    min_radius = max(2, int(round(pitch * 0.20)))
    max_radius = max(min_radius + 1, int(round(pitch * 0.62)))
    support: list[_PadCandidate] = []
    for candidate in inside:
        if candidate.radius < min_radius or candidate.radius > max_radius:
            continue
        if not _roi_contains_circle_center_band(
            roi=roi,
            x=candidate.x,
            y=candidate.y,
            radius=candidate.radius,
        ):
            continue

        evidence_score = _grid_slot_evidence_score(
            image=image,
            x=candidate.x,
            y=candidate.y,
            radius=max(seed_radius, candidate.radius),
        )
        has_neighbor = _has_pitch_neighbor(
            candidate=candidate,
            candidates=inside,
            pitch=pitch,
            config=config,
        )
        if evidence_score < config.final_grid_candidate_min_score and not has_neighbor:
            continue

        support.append(
            _PadCandidate(
                x=candidate.x,
                y=candidate.y,
                radius=candidate.radius,
                score=candidate.score + evidence_score,
                source=candidate.source,
            ),
        )

    return _merge_pad_candidates(support, config)


def _roi_contains_circle_center_band(
    roi: RoiBounds,
    x: int,
    y: int,
    radius: int,
) -> bool:
    """Reject edge candidates whose contour would sit mostly outside the ROI."""
    inset = max(2, int(round(radius * 0.35)))
    return (
        roi.x_min + inset <= x <= roi.x_max - inset
        and roi.y_min + inset <= y <= roi.y_max - inset
    )


def _fit_final_roi_axis_grid(
    image: np.ndarray,
    candidates: list[_PadCandidate],
    roi: RoiBounds,
    pitch: float,
    config: HoughCircleConfig,
) -> _BgaGridFit | None:
    """Fit an ROI-bounded 16x16 grid with separate X/Y projected pitch."""
    expected_slots = config.expected_grid_rows * config.expected_grid_cols
    if len(candidates) < max(30, int(round(expected_slots * 0.35))):
        return None

    points = np.array([(candidate.x, candidate.y) for candidate in candidates], dtype=np.float32)
    x_sets = _axis_position_sets_for_roi(
        values=points[:, 0],
        roi_min=float(roi.x_min),
        roi_max=float(roi.x_max),
        line_count=config.expected_grid_cols,
        axis=0,
        points=points,
        fallback_pitch=pitch,
        config=config,
    )
    y_sets = _axis_position_sets_for_roi(
        values=points[:, 1],
        roi_min=float(roi.y_min),
        roi_max=float(roi.y_max),
        line_count=config.expected_grid_rows,
        axis=1,
        points=points,
        fallback_pitch=pitch,
        config=config,
    )
    if not x_sets or not y_sets:
        return None

    best: _BgaGridFit | None = None
    candidate_circles = [candidate.circle for candidate in candidates]
    for x_score, x_positions in x_sets:
        for y_score, y_positions in y_sets:
            pitch_x = _median_axis_pitch(list(x_positions), fallback=pitch)
            pitch_y = _median_axis_pitch(list(y_positions), fallback=pitch)
            grid = _BgaGridFit(
                score=0.0,
                occupied=0,
                pitch_x=pitch_x,
                pitch_y=pitch_y,
                x_positions=x_positions,
                y_positions=y_positions,
            )
            assignments = _assign_candidates_to_axis_grid(
                candidate_circles,
                grid,
                config,
                tolerance_ratio=config.final_grid_assign_tolerance_ratio,
            )
            occupied = len(assignments)
            span_penalty = _axis_grid_roi_span_penalty(grid=grid, roi=roi)
            line_balance_penalty = _axis_grid_line_balance_penalty(
                candidates=candidate_circles,
                grid=grid,
                config=config,
            )
            slot_evidence_score = _axis_grid_slot_evidence_score(
                image=image,
                grid=grid,
                roi=roi,
                config=config,
            )
            score = (
                (occupied * 100.0)
                + slot_evidence_score
                + x_score
                + y_score
                - span_penalty
                - line_balance_penalty
            )
            if best is None or score > best.score:
                best = _BgaGridFit(
                    score=score,
                    occupied=occupied,
                    pitch_x=pitch_x,
                    pitch_y=pitch_y,
                    x_positions=x_positions,
                    y_positions=y_positions,
                )

    if best is None:
        return None

    return _refine_final_axis_grid(
        candidates=candidates,
        grid=best,
        config=config,
    )


def _axis_position_sets_for_roi(
    values: np.ndarray,
    roi_min: float,
    roi_max: float,
    line_count: int,
    axis: int,
    points: np.ndarray,
    fallback_pitch: float,
    config: HoughCircleConfig,
) -> list[tuple[float, tuple[float, ...]]]:
    """Build candidate line sets from ROI geometry and detected pad centers."""
    pitches = _axis_pitch_candidates_from_points(
        points=points,
        axis=axis,
        roi_min=roi_min,
        roi_max=roi_max,
        line_count=line_count,
        config=config,
    )
    roi_span = max(1.0, roi_max - roi_min)
    for margin_ratio in (0.45, 0.60, 0.75, 1.00, 1.25):
        pitches.append(roi_span / float((line_count - 1) + (2.0 * margin_ratio)))
    if fallback_pitch > 0:
        pitches.append(float(fallback_pitch))

    unique_pitches: list[float] = []
    for candidate_pitch in sorted(pitches):
        if candidate_pitch <= 0:
            continue
        if all(abs(candidate_pitch - existing) > max(2.0, existing * 0.08) for existing in unique_pitches):
            unique_pitches.append(candidate_pitch)

    scored_sets: list[tuple[float, tuple[float, ...]]] = []
    scored_sets.extend(
        _clustered_axis_position_sets(
            values=values,
            line_count=line_count,
            pitches=unique_pitches,
            config=config,
        ),
    )
    for candidate_pitch in unique_pitches:
        scored_sets.extend(
            _best_axis_position_sets(
                values=values,
                pitch=candidate_pitch,
                line_count=line_count,
                keep=config.final_grid_axis_keep_count,
                config=config,
            ),
        )
        for margin_ratio in (0.45, 0.60, 0.75, 1.00, 1.25):
            positions = tuple(
                roi_min
                + (candidate_pitch * margin_ratio)
                + (index * candidate_pitch)
                for index in range(line_count)
            )
            boundary_slack = candidate_pitch * 0.35
            if positions[0] < roi_min - boundary_slack or positions[-1] > roi_max + boundary_slack:
                continue
            scored_sets.append(
                (
                    _axis_position_score(values, positions, candidate_pitch, config),
                    positions,
                ),
            )

    return _deduplicate_axis_position_sets(
        scored_sets=scored_sets,
        keep=config.final_grid_axis_keep_count,
    )


def _clustered_axis_position_sets(
    values: np.ndarray,
    line_count: int,
    pitches: list[float],
    config: HoughCircleConfig,
) -> list[tuple[float, tuple[float, ...]]]:
    """Build NON-uniform line sets directly from candidate coordinate clusters.

    Oblique X-ray views foreshorten one axis, so the real line spacing varies
    across the package (perspective) and no uniform-pitch series can match
    every row/column. Clustering the observed pad coordinates recovers the
    true, gradually changing line positions.
    """
    if values.size == 0:
        return []

    sorted_values = np.sort(values.astype(np.float64))
    scored_sets: list[tuple[float, tuple[float, ...]]] = []

    for pitch in pitches:
        if pitch <= 0:
            continue

        break_gap = max(8.0, pitch * 0.45)
        clusters: list[list[float]] = [[float(sorted_values[0])]]
        for value in sorted_values[1:]:
            if float(value) - clusters[-1][-1] <= break_gap:
                clusters[-1].append(float(value))
            else:
                clusters.append([float(value)])

        strong = [cluster for cluster in clusters if len(cluster) >= 3]
        if len(strong) < line_count:
            continue

        positions_all = [float(np.median(cluster)) for cluster in strong]
        supports = [len(cluster) for cluster in strong]

        # Slide a window of line_count consecutive clusters and keep the
        # windows with the highest support and plausible spacing.
        for start in range(0, len(strong) - line_count + 1):
            window_positions = positions_all[start : start + line_count]
            gaps = np.diff(window_positions)
            if np.any(gaps <= 0):
                continue

            median_gap = float(np.median(gaps))
            if median_gap <= 0:
                continue
            # The window must match the hypothesis pitch scale: without this,
            # dense junk clusters (vias, traces) form uniformly spaced but
            # absurdly narrow line sets that outscore the real grid.
            if median_gap < pitch * 0.65 or median_gap > pitch * 1.45:
                continue
            ratios = gaps / median_gap
            if np.any(ratios < 0.55) or np.any(ratios > 1.80):
                continue

            positions = tuple(window_positions)
            score = _axis_position_score(values, positions, median_gap, config)
            # Small bonus: real cluster lines carry direct support.
            score += float(sum(supports[start : start + line_count])) * 0.05
            scored_sets.append((score, positions))

    return scored_sets


def _axis_pitch_candidates_from_points(
    points: np.ndarray,
    axis: int,
    roi_min: float,
    roi_max: float,
    line_count: int,
    config: HoughCircleConfig,
) -> list[float]:
    """Estimate pitch for one projected axis from near-row/near-column gaps."""
    if len(points) < 2:
        return []

    roi_span = max(1.0, roi_max - roi_min)
    min_pitch = max(18.0, roi_span / float(line_count * 1.75))
    max_pitch = min(config.grid_search_max_pitch, roi_span / max(1.0, line_count * 0.42))
    if max_pitch <= min_pitch:
        max_pitch = min_pitch * 1.8

    sampled_points = points
    if len(sampled_points) > 700:
        indexes = np.linspace(0, len(sampled_points) - 1, 700).astype(int)
        sampled_points = sampled_points[indexes]

    gap_values: list[float] = []
    other_axis = 1 - axis
    for tolerance in (10.0, 14.0, 18.0, 24.0, 32.0, 42.0):
        for point in sampled_points:
            axis_distance = np.abs(sampled_points[:, axis] - point[axis])
            other_distance = np.abs(sampled_points[:, other_axis] - point[other_axis])
            gaps = axis_distance[
                (other_distance <= tolerance)
                & (axis_distance >= min_pitch)
                & (axis_distance <= max_pitch)
            ]
            gap_values.extend(gaps.tolist())

    if not gap_values:
        return []

    bin_width = config.grid_pitch_histogram_bin
    bins = np.arange(min_pitch, max_pitch + bin_width, bin_width)
    if bins.size < 2:
        return []

    histogram, edges = np.histogram(gap_values, bins=bins)
    pitches: list[float] = []
    for index in np.argsort(histogram)[-8:][::-1]:
        if histogram[index] < 4:
            continue
        candidate_pitch = float((edges[index] + edges[index + 1]) / 2.0)
        if all(abs(candidate_pitch - existing) > bin_width * 2.0 for existing in pitches):
            pitches.append(candidate_pitch)

    return pitches


def _axis_position_score(
    values: np.ndarray,
    positions: tuple[float, ...],
    pitch: float,
    config: HoughCircleConfig,
) -> float:
    """Score how well one axis-line set covers candidate coordinates."""
    tolerance = max(10.0, pitch * config.grid_line_tolerance_ratio)
    position_array = np.array(positions, dtype=np.float32)
    distances = np.abs(values[:, None] - position_array[None, :])
    counts = np.count_nonzero(distances <= tolerance, axis=0)
    occupied_lines = int(np.count_nonzero(counts > 0))
    capped_count = int(np.sum(np.minimum(counts, len(positions))))
    overfull_limit = max(4, int(round(len(positions) * 1.35)))
    overfull_penalty = int(np.sum(np.maximum(counts - overfull_limit, 0))) * 4
    edge_bonus = 2 if counts[0] > 0 and counts[-1] > 0 else 0
    return float((occupied_lines * 10.0) + capped_count + edge_bonus - overfull_penalty)


def _deduplicate_axis_position_sets(
    scored_sets: list[tuple[float, tuple[float, ...]]],
    keep: int,
) -> list[tuple[float, tuple[float, ...]]]:
    """Keep the strongest distinct axis position sets."""
    scored_sets.sort(key=lambda item: item[0], reverse=True)
    unique: list[tuple[float, tuple[float, ...]]] = []
    for score, positions in scored_sets:
        pitch = _median_axis_pitch(list(positions), fallback=0.0)
        if pitch <= 0:
            continue
        if any(
            abs(positions[0] - existing_positions[0]) < pitch * 0.18
            and abs(pitch - _median_axis_pitch(list(existing_positions), fallback=pitch))
            < pitch * 0.10
            for _, existing_positions in unique
        ):
            continue
        unique.append((score, positions))
        if len(unique) >= keep:
            break

    return unique


def _edge_assignment_count(
    assignments: dict[tuple[int, int], int],
    config: HoughCircleConfig,
) -> int:
    """Count assignments on the outer rows and columns."""
    last_row = config.expected_grid_rows - 1
    last_column = config.expected_grid_cols - 1
    return sum(
        1
        for row, column in assignments
        if row in (0, last_row) or column in (0, last_column)
    )


def _axis_grid_line_balance_penalty(
    candidates: list[tuple[int, int, int]],
    grid: _BgaGridFit,
    config: HoughCircleConfig,
) -> float:
    """Penalize projected grid lines that absorb multiple physical rows/columns."""
    if not candidates:
        return 0.0

    points = np.array([(x, y) for x, y, _ in candidates], dtype=np.float32)
    tolerance_x = max(10.0, grid.pitch_x * config.final_grid_assign_tolerance_ratio)
    tolerance_y = max(10.0, grid.pitch_y * config.final_grid_assign_tolerance_ratio)
    expected_rows = config.expected_grid_rows
    expected_cols = config.expected_grid_cols

    row_counts = [
        int(np.count_nonzero(np.abs(points[:, 1] - y_position) <= tolerance_y))
        for y_position in grid.y_positions
    ]
    col_counts = [
        int(np.count_nonzero(np.abs(points[:, 0] - x_position) <= tolerance_x))
        for x_position in grid.x_positions
    ]
    row_over_limit = max(4, int(round(expected_cols * 1.35)))
    col_over_limit = max(4, int(round(expected_rows * 1.35)))
    row_under_limit = max(2, int(round(expected_cols * 0.35)))
    col_under_limit = max(2, int(round(expected_rows * 0.35)))

    row_over = sum(max(0, count - row_over_limit) for count in row_counts)
    col_over = sum(max(0, count - col_over_limit) for count in col_counts)
    row_under = sum(max(0, row_under_limit - count) for count in row_counts)
    col_under = sum(max(0, col_under_limit - count) for count in col_counts)
    return float((row_over + col_over) * 36.0 + (row_under + col_under) * 10.0)


def _axis_grid_slot_evidence_score(
    image: np.ndarray,
    grid: _BgaGridFit,
    roi: RoiBounds,
    config: HoughCircleConfig,
) -> float:
    """Score a whole 16x16 grid by checking pad-like evidence at each slot."""
    pitch = float((grid.pitch_x + grid.pitch_y) / 2.0)
    seed_radius = max(
        config.grid_radius_min,
        int(round(pitch * config.roi_pad_radius_seed_pitch_ratio)),
    )
    scores: list[float] = []
    for y_position in grid.y_positions:
        for x_position in grid.x_positions:
            x = int(round(x_position))
            y = int(round(y_position))
            if not _roi_contains_circle_center_band(
                roi=roi,
                x=x,
                y=y,
                radius=seed_radius,
            ):
                scores.append(0.0)
                continue
            scores.append(_grid_slot_evidence_score(image, x, y, seed_radius))

    if not scores:
        return 0.0

    score_array = np.array(scores, dtype=np.float32)
    occupied = int(np.count_nonzero(score_array >= 0.34))
    strong = int(np.count_nonzero(score_array >= 0.58))
    mean_score = float(np.mean(score_array))
    low_penalty = float(np.count_nonzero(score_array < 0.24)) * 18.0
    return (occupied * 16.0) + (strong * 10.0) + (mean_score * 256.0) - low_penalty


def _axis_grid_roi_span_penalty(grid: _BgaGridFit, roi: RoiBounds) -> float:
    """Penalize grids whose projected lines extend too far outside the ROI."""
    overflow = 0.0
    overflow += max(0.0, float(roi.x_min) - grid.x_positions[0])
    overflow += max(0.0, grid.x_positions[-1] - float(roi.x_max))
    overflow += max(0.0, float(roi.y_min) - grid.y_positions[0])
    overflow += max(0.0, grid.y_positions[-1] - float(roi.y_max))
    return overflow * 8.0


def _refine_final_axis_grid(
    candidates: list[_PadCandidate],
    grid: _BgaGridFit,
    config: HoughCircleConfig,
) -> _BgaGridFit:
    """Nudge axis-grid lines toward assigned candidate medians."""
    candidate_circles = [candidate.circle for candidate in candidates]
    refined = grid
    min_column_support = max(
        config.grid_refine_min_axis_assignments,
        int(round(config.expected_grid_rows * config.final_grid_refine_min_axis_fraction)),
    )
    min_row_support = max(
        config.grid_refine_min_axis_assignments,
        int(round(config.expected_grid_cols * config.final_grid_refine_min_axis_fraction)),
    )
    for _ in range(2):
        assignments = _assign_candidates_to_axis_grid(
            candidate_circles,
            refined,
            config,
            tolerance_ratio=config.final_grid_assign_tolerance_ratio,
        )
        x_positions = list(refined.x_positions)
        y_positions = list(refined.y_positions)
        for column in range(config.expected_grid_cols):
            column_values = [
                float(candidates[index].x)
                for (row, assigned_column), index in assignments.items()
                if assigned_column == column
            ]
            if len(column_values) >= min_column_support:
                x_positions[column] = float(np.median(column_values))

        for row in range(config.expected_grid_rows):
            row_values = [
                float(candidates[index].y)
                for (assigned_row, column), index in assignments.items()
                if assigned_row == row
            ]
            if len(row_values) >= min_row_support:
                y_positions[row] = float(np.median(row_values))

        pitch_x = _median_axis_pitch(x_positions, fallback=refined.pitch_x)
        pitch_y = _median_axis_pitch(y_positions, fallback=refined.pitch_y)
        refined = _BgaGridFit(
            score=refined.score,
            occupied=len(assignments),
            pitch_x=pitch_x,
            pitch_y=pitch_y,
            x_positions=tuple(x_positions),
            y_positions=tuple(y_positions),
        )

    return refined


def _stabilize_axis_positions(
    positions: list[float],
    fallback_pitch: float,
    config: HoughCircleConfig,
) -> list[float]:
    """Prevent refined grid axes from collapsing onto neighboring rows/columns."""
    if len(positions) < 2 or fallback_pitch <= 0.0:
        return positions

    gaps = np.diff(np.array(positions, dtype=np.float32))
    positive_gaps = gaps[gaps > 0]
    if positive_gaps.size == 0:
        return positions

    pitch_candidates = positive_gaps[
        (positive_gaps >= fallback_pitch * 0.55)
        & (positive_gaps <= fallback_pitch * 1.45)
    ]
    reference_pitch = (
        float(np.median(pitch_candidates))
        if pitch_candidates.size
        else fallback_pitch
    )
    min_gap = reference_pitch * config.final_grid_axis_min_gap_ratio
    max_gap = reference_pitch * config.final_grid_axis_max_gap_ratio
    if np.all((gaps >= min_gap) & (gaps <= max_gap)):
        return positions

    indexes = np.arange(len(positions), dtype=np.float32)
    starts = np.array(positions, dtype=np.float32) - (indexes * reference_pitch)
    start = float(np.median(starts))
    return [start + (index * reference_pitch) for index in range(len(positions))]


def _pad_candidates_from_axis_grid_assignments(
    image: np.ndarray,
    candidates: list[_PadCandidate],
    grid: _BgaGridFit,
    assignments: dict[tuple[int, int], int],
    config: HoughCircleConfig,
) -> list[_PadCandidate]:
    """Return one final pad candidate per assigned grid slot."""
    pitch = float((grid.pitch_x + grid.pitch_y) / 2.0)
    seed_radius = max(
        config.grid_radius_min,
        int(round(pitch * config.roi_pad_radius_seed_pitch_ratio)),
    )
    regularized: list[_PadCandidate] = []
    for row, column in sorted(assignments):
        candidate = candidates[assignments[(row, column)]]
        slot_x = int(round(grid.x_positions[column]))
        slot_y = int(round(grid.y_positions[row]))
        center_x = slot_x if config.snap_final_candidates_to_grid else candidate.x
        center_y = slot_y if config.snap_final_candidates_to_grid else candidate.y
        candidate_score = _grid_slot_evidence_score(
            image=image,
            x=center_x,
            y=center_y,
            radius=max(seed_radius, candidate.radius),
        )

        regularized.append(
            _PadCandidate(
                x=center_x,
                y=center_y,
                radius=candidate.radius,
                score=candidate.score + candidate_score,
                source=candidate.source,
            ),
        )

    return _merge_pad_candidates(regularized, config)


def _fit_axis_cluster_grid_from_candidates(
    candidates: list[_PadCandidate],
    pitch: float,
    config: HoughCircleConfig,
) -> _BgaGridFit | None:
    """Fit a 16x16 grid with independent X/Y spacing from candidate clusters."""
    expected_slots = config.expected_grid_rows * config.expected_grid_cols
    if len(candidates) < int(expected_slots * config.roi_component_candidate_min_result_ratio):
        return None

    tolerance = max(14.0, pitch * config.roi_component_axis_cluster_tolerance_ratio)
    x_clusters = _cluster_pad_candidates_by_coordinate(
        candidates=candidates,
        coordinate_getter=lambda candidate: candidate.x,
        tolerance=tolerance,
    )
    y_clusters = _cluster_pad_candidates_by_coordinate(
        candidates=candidates,
        coordinate_getter=lambda candidate: candidate.y,
        tolerance=tolerance,
    )
    if (
        len(x_clusters) != config.expected_grid_cols
        or len(y_clusters) != config.expected_grid_rows
    ):
        return None

    x_positions = tuple(
        float(np.median([candidate.x for candidate in cluster]))
        for cluster in x_clusters
    )
    y_positions = tuple(
        float(np.median([candidate.y for candidate in cluster]))
        for cluster in y_clusters
    )
    pitch_x = _median_axis_pitch(list(x_positions), fallback=pitch)
    pitch_y = _median_axis_pitch(list(y_positions), fallback=pitch)
    assignments = _assign_candidates_to_axis_grid(
        [candidate.circle for candidate in candidates],
        _BgaGridFit(
            score=0.0,
            occupied=0,
            pitch_x=pitch_x,
            pitch_y=pitch_y,
            x_positions=x_positions,
            y_positions=y_positions,
        ),
        config,
        tolerance_ratio=config.roi_component_axis_assign_tolerance_ratio,
    )
    return _BgaGridFit(
        score=float(len(assignments)),
        occupied=len(assignments),
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        x_positions=x_positions,
        y_positions=y_positions,
    )


def _assign_candidates_to_axis_grid(
    candidates: list[tuple[int, int, int]],
    grid: _BgaGridFit,
    config: HoughCircleConfig,
    tolerance_ratio: float,
) -> dict[tuple[int, int], int]:
    """Assign candidates using independent X/Y tolerances for projected grids."""
    if not candidates:
        return {}

    points = np.array([(x, y) for x, y, _ in candidates], dtype=np.float32)
    tolerance_x = max(10.0, grid.pitch_x * tolerance_ratio)
    tolerance_y = max(10.0, grid.pitch_y * tolerance_ratio)
    assignments: dict[tuple[int, int], int] = {}
    used: set[int] = set()
    for row, y in enumerate(grid.y_positions):
        for column, x in enumerate(grid.x_positions):
            dx = np.abs(points[:, 0] - x)
            dy = np.abs(points[:, 1] - y)
            normalized_distance = (dx / tolerance_x) ** 2 + (dy / tolerance_y) ** 2
            for candidate_index in np.argsort(normalized_distance):
                index = int(candidate_index)
                if index in used:
                    continue
                if dx[index] > tolerance_x or dy[index] > tolerance_y:
                    break
                assignments[(row, column)] = index
                used.add(index)
                break

    return assignments


def _assign_pad_candidates_to_axis_grid(
    image: np.ndarray,
    candidates: list[_PadCandidate],
    grid: _BgaGridFit,
    config: HoughCircleConfig,
    tolerance_ratio: float,
) -> dict[tuple[int, int], int]:
    """Assign one pad candidate per grid slot using geometry plus local pad evidence."""
    if not config.final_grid_use_evidence_weighted_assignment:
        return _assign_candidates_to_axis_grid(
            [candidate.circle for candidate in candidates],
            grid,
            config,
            tolerance_ratio=tolerance_ratio,
        )
    if not candidates:
        return {}

    points = np.array([(candidate.x, candidate.y) for candidate in candidates], dtype=np.float32)
    tolerance_x = max(10.0, grid.pitch_x * tolerance_ratio)
    tolerance_y = max(10.0, grid.pitch_y * tolerance_ratio)
    pitch = float((grid.pitch_x + grid.pitch_y) / 2.0)
    seed_radius = max(
        config.grid_radius_min,
        int(round(pitch * config.roi_pad_radius_seed_pitch_ratio)),
    )
    last_row = config.expected_grid_rows - 1
    last_column = config.expected_grid_cols - 1
    scored_pairs: list[tuple[float, tuple[int, int], int]] = []

    for row, y in enumerate(grid.y_positions):
        for column, x in enumerate(grid.x_positions):
            dx = np.abs(points[:, 0] - x)
            dy = np.abs(points[:, 1] - y)
            within_tolerance = np.where((dx <= tolerance_x) & (dy <= tolerance_y))[0]
            if within_tolerance.size == 0:
                continue

            min_evidence = config.final_grid_assignment_min_evidence
            if row in (0, last_row) or column in (0, last_column):
                min_evidence = config.final_grid_assignment_edge_min_evidence

            for candidate_index_np in within_tolerance:
                candidate_index = int(candidate_index_np)
                candidate = candidates[candidate_index]
                radius = max(seed_radius, candidate.radius)
                evidence_score = _grid_slot_evidence_score(
                    image=image,
                    x=candidate.x,
                    y=candidate.y,
                    radius=radius,
                )
                if evidence_score < min_evidence:
                    continue

                normalized_distance = (
                    (float(dx[candidate_index]) / tolerance_x) ** 2
                    + (float(dy[candidate_index]) / tolerance_y) ** 2
                )
                candidate_score = min(5.0, max(0.0, candidate.score)) / 5.0
                pair_score = (
                    evidence_score * 3.0
                    + candidate_score * config.final_grid_assignment_candidate_weight
                    - normalized_distance * config.final_grid_assignment_distance_weight
                )
                scored_pairs.append((pair_score, (row, column), candidate_index))

    scored_pairs.sort(key=lambda item: item[0], reverse=True)
    assignments: dict[tuple[int, int], int] = {}
    used_slots: set[tuple[int, int]] = set()
    used_candidates: set[int] = set()
    for _, slot, candidate_index in scored_pairs:
        if slot in used_slots or candidate_index in used_candidates:
            continue
        assignments[slot] = candidate_index
        used_slots.add(slot)
        used_candidates.add(candidate_index)

    return assignments


def _refine_roi_from_dominant_component(
    candidates: list[_PadCandidate],
    initial_roi: RoiBounds | None,
    image_shape: tuple[int, int],
    pitch: float | None,
    config: HoughCircleConfig,
) -> RoiBounds | None:
    """Tighten the debug ROI around the dominant pitch-connected BGA component."""
    if initial_roi is None or pitch is None or pitch <= 0.0:
        return initial_roi

    expected_slots = config.expected_grid_rows * config.expected_grid_cols
    min_component_size = int(expected_slots * config.roi_component_refine_min_ratio)
    if len(candidates) < min_component_size:
        return initial_roi

    points = np.array([(candidate.x, candidate.y) for candidate in candidates], dtype=np.float32)
    component_indexes = _largest_grid_neighbor_component(points, pitch, config)
    if len(component_indexes) < min_component_size:
        return initial_roi

    component_points = points[component_indexes]
    x0, y0 = np.min(component_points, axis=0)
    x1, y1 = np.max(component_points, axis=0)
    center_x = float((x0 + x1) / 2.0)
    center_y = float((y0 + y1) / 2.0)
    observed_width = float(x1 - x0)
    observed_height = float(y1 - y0)
    expected_width = float(
        (config.expected_grid_cols - 1)
        * pitch
        * config.roi_component_expected_span_min_ratio,
    )
    expected_height = float(
        (config.expected_grid_rows - 1)
        * pitch
        * config.roi_component_expected_span_min_ratio,
    )
    grid_width = max(observed_width, expected_width)
    grid_height = max(observed_height, expected_height)
    margin = pitch * config.roi_component_margin_pitch_ratio

    refined = _clamp_roi(
        RoiBounds(
            x_min=int(round(center_x - (grid_width / 2.0) - margin)),
            y_min=int(round(center_y - (grid_height / 2.0) - margin)),
            x_max=int(round(center_x + (grid_width / 2.0) + margin)),
            y_max=int(round(center_y + (grid_height / 2.0) + margin)),
        ),
        image_shape,
    )
    refined_area = float((refined.x_max - refined.x_min) * (refined.y_max - refined.y_min))
    initial_area = float(
        (initial_roi.x_max - initial_roi.x_min)
        * (initial_roi.y_max - initial_roi.y_min),
    )
    if refined_area <= 0.0 or initial_area <= 0.0:
        return initial_roi

    # Keep the initial ROI when the refinement would only expand the debug box.
    if refined_area >= initial_area * 1.08:
        return initial_roi

    refined = _trim_roi_to_dense_candidate_rows(
        candidates=candidates,
        roi=refined,
        pitch=pitch,
        image_shape=image_shape,
        config=config,
    )
    return refined


def _trim_roi_to_dense_candidate_rows(
    candidates: list[_PadCandidate],
    roi: RoiBounds,
    pitch: float,
    image_shape: tuple[int, int],
    config: HoughCircleConfig,
) -> RoiBounds:
    """Trim sparse extreme row clusters from the BGA ROI."""
    inside = [
        candidate
        for candidate in candidates
        if _roi_contains_point(roi, candidate.x, candidate.y)
    ]
    if len(inside) < int(config.expected_grid_cols * config.expected_grid_rows * 0.50):
        return roi

    tolerance = max(12.0, pitch * config.roi_dense_row_tolerance_ratio)
    clusters = _cluster_pad_candidates_by_coordinate(
        candidates=inside,
        coordinate_getter=lambda candidate: candidate.y,
        tolerance=tolerance,
    )
    if len(clusters) < int(config.expected_grid_rows * 0.75):
        return roi

    cluster_counts = np.array([len(cluster) for cluster in clusters], dtype=np.float32)
    reference_count = float(np.median(cluster_counts))
    min_count = max(
        config.roi_dense_row_min_count,
        int(round(reference_count * config.roi_dense_row_min_fraction)),
    )
    start = 0
    end = len(clusters) - 1
    while start <= end and len(clusters[start]) < min_count:
        start += 1
    while end >= start and len(clusters[end]) < min_count:
        end -= 1
    kept_clusters = clusters[start : end + 1]
    if len(kept_clusters) < int(config.expected_grid_rows * 0.75):
        return roi
    if start == 0 and end == len(clusters) - 1:
        return roi

    row_centers = [
        float(np.median([candidate.y for candidate in cluster]))
        for cluster in kept_clusters
    ]
    margin = pitch * config.roi_dense_row_margin_pitch_ratio
    return _clamp_roi(
        RoiBounds(
            x_min=roi.x_min,
            y_min=int(round(min(row_centers) - margin)),
            x_max=roi.x_max,
            y_max=int(round(max(row_centers) + margin)),
        ),
        image_shape,
    )


def _cluster_pad_candidates_by_coordinate(
    candidates: list[_PadCandidate],
    coordinate_getter: Callable[[_PadCandidate], int],
    tolerance: float,
) -> list[list[_PadCandidate]]:
    """Cluster pad candidates by a single coordinate."""
    clusters: list[list[_PadCandidate]] = []
    cluster_values: list[list[float]] = []
    for candidate in sorted(candidates, key=coordinate_getter):
        value = float(coordinate_getter(candidate))
        if cluster_values and abs(value - float(np.mean(cluster_values[-1]))) <= tolerance:
            clusters[-1].append(candidate)
            cluster_values[-1].append(value)
        else:
            clusters.append([candidate])
            cluster_values.append([value])
    return clusters


def _is_near_anchor_candidate(
    candidate: _PadCandidate,
    anchors: list[_PadCandidate],
    max_distance: float,
) -> bool:
    """Return true when a candidate refines an existing large pad anchor."""
    return any(
        np.hypot(candidate.x - anchor.x, candidate.y - anchor.y) <= max_distance
        for anchor in anchors
    )


def _filter_bga_roi_pad_candidates(
    image: np.ndarray,
    candidates: list[_PadCandidate],
    roi_estimate: _BgaRoiEstimate,
    config: HoughCircleConfig,
) -> tuple[list[_PadCandidate], list[MissingGridPosition], int]:
    """Keep real pad candidates inside the BGA ROI and reject grid outliers."""
    roi = roi_estimate.roi
    pitch = roi_estimate.pitch
    inside = [
        candidate
        for candidate in candidates
        if _roi_contains_point(roi, candidate.x, candidate.y)
    ]
    if not inside:
        return [], [], 0

    radii = np.array([candidate.radius for candidate in inside], dtype=np.float32)
    reference_radius = float(np.percentile(radii, 65))
    radius_floor = max(
        float(reference_radius * 0.62),
        float(np.percentile(radii, 20) * 0.70),
    )
    radius_ceiling = float(np.percentile(radii, 95) * 1.30)

    filtered: list[_PadCandidate] = []
    seed_radius = max(
        1,
        int(round(pitch * config.roi_pad_radius_seed_pitch_ratio)),
    )
    for candidate in inside:
        if candidate.radius < radius_floor or candidate.radius > radius_ceiling:
            continue

        evidence_score = _grid_slot_evidence_score(
            image=image,
            x=candidate.x,
            y=candidate.y,
            radius=max(seed_radius, candidate.radius),
        )
        if (
            evidence_score < config.roi_candidate_min_score
            and not _has_pitch_neighbor(candidate, inside, pitch, config)
            and candidate.source != "hough"
        ):
            continue

        filtered.append(
            _PadCandidate(
                x=candidate.x,
                y=candidate.y,
                radius=candidate.radius,
                score=candidate.score + evidence_score,
                source=candidate.source,
            ),
        )

    filtered = _merge_pad_candidates(filtered, config)
    if not filtered:
        return [], [], 0

    grid = _fit_bga_grid([candidate.circle for candidate in filtered], config)
    missing_positions: list[MissingGridPosition] = []
    occupied_slots = len(filtered)
    if grid is not None:
        assignments = _assign_candidates_to_grid(
            [candidate.circle for candidate in filtered],
            grid,
            config,
        )
        occupied_slots = len(assignments)
        expected_slots = config.expected_grid_rows * config.expected_grid_cols
        strong_grid_support = len(assignments) >= int(expected_slots * 0.78)
        if strong_grid_support:
            used_indexes = set(assignments.values())
            filtered = [
                candidate
                for index, candidate in enumerate(filtered)
                if index in used_indexes
            ]
            if config.recover_grid_evidence_candidates:
                recovered_candidates = _recover_grid_evidence_candidates(
                    image=image,
                    grid=grid,
                    assignments=assignments,
                    config=config,
                )
                max_recovered = int(
                    round(expected_slots * config.grid_evidence_max_recovered_ratio),
                )
                if 0 < len(recovered_candidates) <= max_recovered:
                    filtered = _merge_pad_candidates(
                        [*filtered, *recovered_candidates],
                        config,
                    )
                    assignments = _assign_candidates_to_grid(
                        [candidate.circle for candidate in filtered],
                        grid,
                        config,
                    )
            occupied_slots = len(filtered)
            missing_positions = _missing_positions_from_grid(
                grid=grid,
                assignments=assignments,
                config=config,
            )
        elif len(filtered) > expected_slots:
            filtered = _best_neighbor_consistent_candidates(
                candidates=filtered,
                pitch=pitch,
                keep_count=expected_slots,
                config=config,
            )
            occupied_slots = min(len(filtered), expected_slots)

    filtered = _largest_bga_component_candidates(
        candidates=filtered,
        pitch=pitch,
        config=config,
    )
    occupied_slots = len(filtered)
    return filtered, missing_positions, occupied_slots


def _recover_grid_evidence_candidates(
    image: np.ndarray,
    grid: _BgaGridFit,
    assignments: dict[tuple[int, int], int],
    config: HoughCircleConfig,
    min_score: float | None = None,
) -> list[_PadCandidate]:
    """Recover missing grid slots only when local image evidence is strong."""
    score_floor = (
        config.grid_evidence_min_score
        if min_score is None
        else min_score
    )
    pitch = float((grid.pitch_x + grid.pitch_y) / 2.0)
    seed_radius = max(
        config.grid_radius_min,
        int(round(pitch * config.grid_radius_pitch_ratio)),
    )
    min_radius = int(round(pitch * config.roi_pad_radius_min_pitch_ratio))
    max_radius = int(round(pitch * config.roi_pad_radius_max_pitch_ratio))
    recovered: list[_PadCandidate] = []
    for row, y in enumerate(grid.y_positions):
        for column, x in enumerate(grid.x_positions):
            if (row, column) in assignments:
                continue

            center_x = int(round(x))
            center_y = int(round(y))
            evidence_score = _grid_slot_evidence_score(
                image=image,
                x=center_x,
                y=center_y,
                radius=seed_radius,
            )
            if evidence_score < score_floor:
                continue

            metrics = _dark_component_geometry_metrics(
                image=image,
                x=center_x,
                y=center_y,
                radius=seed_radius,
                config=config,
                intensity_threshold=155,
            )
            if metrics is None:
                continue

            aspect_ratio, area_ratio, circularity = metrics
            if area_ratio < 0.16:
                continue
            if aspect_ratio > 3.60:
                continue
            if aspect_ratio > 2.20 and circularity < 0.18:
                continue

            radius = _estimate_pad_radius(
                image=image,
                x=center_x,
                y=center_y,
                radius=seed_radius,
                config=config,
            )
            radius = max(min_radius, min(radius, max_radius))
            recovered.append(
                _PadCandidate(
                    x=center_x,
                    y=center_y,
                    radius=radius,
                    score=1.6 + evidence_score,
                    source="grid-evidence",
                ),
            )

    return recovered


def _recover_slot_local_pad_candidates(
    image: np.ndarray,
    roi: RoiBounds,
    grid: _BgaGridFit,
    assignments: dict[tuple[int, int], int],
    config: HoughCircleConfig,
    assigned_candidates: list[_PadCandidate],
) -> list[_PadCandidate]:
    """Recover missing grid slots by segmenting a zoomed local ROI crop."""
    pitch = float((grid.pitch_x + grid.pitch_y) / 2.0)
    seed_radius = max(
        config.grid_radius_min,
        int(round(pitch * config.roi_pad_radius_seed_pitch_ratio)),
    )
    min_radius = int(round(pitch * config.roi_pad_radius_min_pitch_ratio))
    max_radius = int(round(pitch * config.roi_pad_radius_max_pitch_ratio))
    search_radius = max(
        seed_radius + 4,
        int(round(pitch * config.final_grid_slot_recovery_search_pitch_ratio)),
    )
    max_offset = max(
        8.0,
        pitch * config.final_grid_slot_recovery_max_offset_ratio,
    )
    zoom = max(
        1.0,
        min(
            config.final_grid_slot_recovery_max_zoom,
            config.final_grid_slot_recovery_zoom_target_radius / max(1.0, seed_radius),
        ),
    )
    height, width = image.shape[:2]
    assigned_points = np.array(
        [(candidate.x, candidate.y) for candidate in assigned_candidates],
        dtype=np.float32,
    )
    recovered: list[_PadCandidate] = []

    for row, y in enumerate(grid.y_positions):
        for column, x in enumerate(grid.x_positions):
            if (row, column) in assignments:
                continue

            expected_x = int(round(x))
            expected_y = int(round(y))
            if not _roi_contains_point(roi, expected_x, expected_y):
                continue
            if not _roi_contains_circle_center_band(
                roi=roi,
                x=expected_x,
                y=expected_y,
                radius=min_radius,
            ):
                continue

            x0 = max(roi.x_min, expected_x - search_radius)
            y0 = max(roi.y_min, expected_y - search_radius)
            x1 = min(roi.x_max, expected_x + search_radius)
            y1 = min(roi.y_max, expected_y + search_radius)
            if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
                continue
            if x1 <= x0 or y1 <= y0:
                continue

            if assigned_points.size:
                assigned_distances = np.hypot(
                    assigned_points[:, 0] - expected_x,
                    assigned_points[:, 1] - expected_y,
                )
                if float(np.min(assigned_distances)) < pitch * 0.05:
                    continue

            local_crop = image[y0 : y1 + 1, x0 : x1 + 1]
            if zoom > 1.01:
                local_crop = cv2.resize(
                    local_crop,
                    None,
                    fx=zoom,
                    fy=zoom,
                    interpolation=cv2.INTER_CUBIC,
                )

            candidate = _best_zoomed_slot_component_candidate(
                image=image,
                crop=local_crop,
                crop_origin=(x0, y0),
                zoom=zoom,
                expected_center=(expected_x, expected_y),
                seed_radius=seed_radius,
                pitch=pitch,
                min_radius=min_radius,
                max_radius=max_radius,
                max_offset=max_offset,
                config=config,
            )
            if candidate is None:
                candidate = _best_zoomed_slot_hough_candidate(
                    image=image,
                    crop=local_crop,
                    crop_origin=(x0, y0),
                    zoom=zoom,
                    expected_center=(expected_x, expected_y),
                    seed_radius=seed_radius,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    max_offset=max_offset,
                    config=config,
                )
            if candidate is None:
                evidence_score = _grid_slot_evidence_score(
                    image=image,
                    x=expected_x,
                    y=expected_y,
                    radius=seed_radius,
                )
                if evidence_score >= config.final_grid_recovery_min_score:
                    radius = _estimate_pad_radius(
                        image=image,
                        x=expected_x,
                        y=expected_y,
                        radius=seed_radius,
                        config=config,
                    )
                    candidate = _PadCandidate(
                        x=expected_x,
                        y=expected_y,
                        radius=max(min_radius, min(radius, max_radius)),
                        score=1.0 + evidence_score,
                        source="grid-evidence",
                    )
            if candidate is not None:
                recovered.append(candidate)

    return _merge_pad_candidates(recovered, config)


def _best_zoomed_slot_component_candidate(
    image: np.ndarray,
    crop: np.ndarray,
    crop_origin: tuple[int, int],
    zoom: float,
    expected_center: tuple[int, int],
    seed_radius: int,
    pitch: float,
    min_radius: int,
    max_radius: int,
    max_offset: float,
    config: HoughCircleConfig,
) -> _PadCandidate | None:
    """Return the strongest dark component near one expected grid slot."""
    if crop.size == 0:
        return None

    blurred = cv2.GaussianBlur(crop, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    kernel_size = max(3, int(round(seed_radius * zoom * 0.16)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    expected_x, expected_y = expected_center
    origin_x, origin_y = crop_origin
    expected_area = pi * ((seed_radius * zoom) ** 2)
    min_area = expected_area * config.final_grid_slot_recovery_min_area_ratio
    max_area = expected_area * config.final_grid_slot_recovery_max_area_ratio
    best: tuple[float, _PadCandidate] | None = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width <= 0 or box_height <= 0:
            continue

        aspect_ratio = max(box_width, box_height) / max(1, min(box_width, box_height))
        if aspect_ratio > config.roi_component_pad_max_aspect_ratio:
            continue

        extent = area / max(1.0, float(box_width * box_height))
        if extent < 0.28:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        circularity = (4.0 * pi * area) / (perimeter**2) if perimeter else 0.0
        if circularity < 0.06 and aspect_ratio > 2.2:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        local_x = float(moments["m10"] / moments["m00"])
        local_y = float(moments["m01"] / moments["m00"])
        center_x = int(round(origin_x + (local_x / zoom)))
        center_y = int(round(origin_y + (local_y / zoom)))
        offset = float(np.hypot(center_x - expected_x, center_y - expected_y))
        if offset > max_offset:
            continue

        evidence_score = _grid_slot_evidence_score(
            image=image,
            x=center_x,
            y=center_y,
            radius=seed_radius,
        )
        if evidence_score < config.final_grid_candidate_min_score:
            continue

        radius = int(round(np.sqrt(area / pi) / zoom))
        radius = max(min_radius, min(radius, max_radius))
        area_ratio = area / max(1.0, expected_area)
        area_score = 1.0 - min(1.0, abs(area_ratio - 0.55) / 0.55)
        offset_score = 1.0 - min(1.0, offset / max(1.0, max_offset))
        shape_score = float(np.clip((extent + circularity) / 2.0, 0.0, 1.0))
        score = (
            (evidence_score * 1.6)
            + (area_score * 0.5)
            + (offset_score * 0.6)
            + (shape_score * 0.4)
        )
        candidate = _PadCandidate(
            x=center_x,
            y=center_y,
            radius=radius,
            score=1.3 + score,
            source="roi-slot",
        )
        if best is None or score > best[0]:
            best = (score, candidate)

    return None if best is None else best[1]


def _best_zoomed_slot_hough_candidate(
    image: np.ndarray,
    crop: np.ndarray,
    crop_origin: tuple[int, int],
    zoom: float,
    expected_center: tuple[int, int],
    seed_radius: int,
    min_radius: int,
    max_radius: int,
    max_offset: float,
    config: HoughCircleConfig,
) -> _PadCandidate | None:
    """Use local Hough detection as a fallback for one missing grid slot."""
    if crop.size == 0:
        return None

    blurred = cv2.GaussianBlur(crop, (7, 7), 1.2)
    min_zoom_radius = max(3, int(round(min_radius * zoom * 0.72)))
    max_zoom_radius = max(min_zoom_radius + 1, int(round(max_radius * zoom * 1.18)))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=max(8.0, seed_radius * zoom * 0.80),
        param1=105.0,
        param2=11.0,
        minRadius=min_zoom_radius,
        maxRadius=max_zoom_radius,
    )
    if circles is None:
        return None

    expected_x, expected_y = expected_center
    origin_x, origin_y = crop_origin
    best: tuple[float, _PadCandidate] | None = None
    for local_x, local_y, local_radius in np.round(circles[0, :]).astype(int):
        center_x = int(round(origin_x + (float(local_x) / zoom)))
        center_y = int(round(origin_y + (float(local_y) / zoom)))
        offset = float(np.hypot(center_x - expected_x, center_y - expected_y))
        if offset > max_offset:
            continue

        radius = int(round(float(local_radius) / zoom))
        radius = max(min_radius, min(radius, max_radius))
        evidence_score = _grid_slot_evidence_score(
            image=image,
            x=center_x,
            y=center_y,
            radius=max(seed_radius, radius),
        )
        if evidence_score < config.final_grid_candidate_min_score:
            continue

        height, width = image.shape[:2]
        if (
            center_x - radius < 0
            or center_y - radius < 0
            or center_x + radius >= width
            or center_y + radius >= height
        ):
            continue
        pad_crop = image[
            center_y - radius : center_y + radius + 1,
            center_x - radius : center_x + radius + 1,
        ]
        yy, xx = np.indices(pad_crop.shape)
        mask = np.hypot(xx - radius, yy - radius) <= radius * 0.92
        pixels = pad_crop[mask]
        if pixels.size == 0 or float(np.mean(pixels)) > 182.0:
            continue

        offset_score = 1.0 - min(1.0, offset / max(1.0, max_offset))
        score = (evidence_score * 1.6) + (offset_score * 0.8)
        candidate = _PadCandidate(
            x=center_x,
            y=center_y,
            radius=radius,
            score=1.2 + score,
            source="roi-slot",
        )
        if best is None or score > best[0]:
            best = (score, candidate)

    return None if best is None else best[1]


def _largest_bga_component_candidates(
    candidates: list[_PadCandidate],
    pitch: float,
    config: HoughCircleConfig,
) -> list[_PadCandidate]:
    """Keep the dominant pitch-connected BGA component."""
    expected_slots = config.expected_grid_rows * config.expected_grid_cols
    if len(candidates) < int(expected_slots * 0.60):
        return candidates

    points = np.array([(candidate.x, candidate.y) for candidate in candidates])
    component_indexes = _largest_grid_neighbor_component(points, pitch, config)
    if len(component_indexes) < int(expected_slots * 0.60):
        return candidates

    kept_indexes = set(component_indexes)
    if len(component_indexes) >= len(candidates) * config.roi_component_keep_full_ratio:
        component_points = points[component_indexes]
        component_roi = _component_roi_from_points(
            points=component_points,
            pitch=pitch,
            config=config,
        )
        return [
            candidate
            for index, candidate in enumerate(candidates)
            if index in kept_indexes
            or _roi_contains_point(component_roi, candidate.x, candidate.y)
        ]

    return [
        candidate
        for index, candidate in enumerate(candidates)
        if index in kept_indexes
    ]


def _component_roi_from_points(
    points: np.ndarray,
    pitch: float,
    config: HoughCircleConfig,
) -> RoiBounds:
    """Build an unclamped ROI around a dominant grid component."""
    x0, y0 = np.min(points, axis=0)
    x1, y1 = np.max(points, axis=0)
    center_x = float((x0 + x1) / 2.0)
    center_y = float((y0 + y1) / 2.0)
    observed_width = float(x1 - x0)
    observed_height = float(y1 - y0)
    expected_width = float(
        (config.expected_grid_cols - 1)
        * pitch
        * config.roi_component_expected_span_min_ratio,
    )
    expected_height = float(
        (config.expected_grid_rows - 1)
        * pitch
        * config.roi_component_expected_span_min_ratio,
    )
    grid_width = max(observed_width, expected_width)
    grid_height = max(observed_height, expected_height)
    margin = pitch * config.roi_component_margin_pitch_ratio
    return RoiBounds(
        x_min=int(round(center_x - (grid_width / 2.0) - margin)),
        y_min=int(round(center_y - (grid_height / 2.0) - margin)),
        x_max=int(round(center_x + (grid_width / 2.0) + margin)),
        y_max=int(round(center_y + (grid_height / 2.0) + margin)),
    )


def _best_neighbor_consistent_candidates(
    candidates: list[_PadCandidate],
    pitch: float,
    keep_count: int,
    config: HoughCircleConfig,
) -> list[_PadCandidate]:
    """Keep the strongest real candidates when grid assignment is unstable."""
    scored: list[tuple[float, _PadCandidate]] = []
    for candidate in candidates:
        neighbor_count = 0
        for other in candidates:
            if other is candidate:
                continue

            dx = abs(candidate.x - other.x)
            dy = abs(candidate.y - other.y)
            row_tolerance = max(12.0, pitch * config.roi_neighbor_tolerance_ratio)
            min_gap = pitch * config.roi_neighbor_min_pitch_ratio
            max_gap = pitch * config.roi_neighbor_max_pitch_ratio
            same_row = dy <= row_tolerance and min_gap <= dx <= max_gap
            same_column = dx <= row_tolerance and min_gap <= dy <= max_gap
            if same_row or same_column:
                neighbor_count += 1

        score = candidate.score + min(neighbor_count, 8) * 0.25
        scored.append((score, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    kept = [candidate for _, candidate in scored[:keep_count]]
    return sorted(kept, key=lambda item: (item.y, item.x))


def _has_pitch_neighbor(
    candidate: _PadCandidate,
    candidates: list[_PadCandidate],
    pitch: float,
    config: HoughCircleConfig,
) -> bool:
    """Return true when a candidate has a plausible row/column BGA neighbor."""
    row_tolerance = max(12.0, pitch * config.roi_neighbor_tolerance_ratio)
    min_gap = pitch * config.roi_neighbor_min_pitch_ratio
    max_gap = pitch * config.roi_neighbor_max_pitch_ratio
    for other in candidates:
        if other is candidate:
            continue

        dx = abs(candidate.x - other.x)
        dy = abs(candidate.y - other.y)
        same_row = dy <= row_tolerance and min_gap <= dx <= max_gap
        same_column = dx <= row_tolerance and min_gap <= dy <= max_gap
        if same_row or same_column:
            return True

    return False


def _missing_positions_from_grid(
    grid: _BgaGridFit,
    assignments: dict[tuple[int, int], int],
    config: HoughCircleConfig,
) -> list[MissingGridPosition]:
    """Report unconfirmed grid slots without creating artificial detections."""
    missing_positions: list[MissingGridPosition] = []
    for row, y in enumerate(grid.y_positions):
        for column, x in enumerate(grid.x_positions):
            if (row, column) in assignments:
                continue

            missing_positions.append(
                MissingGridPosition(
                    row=row + 1,
                    column=column + 1,
                    center_x=int(round(x)),
                    center_y=int(round(y)),
                    reason="expected 16x16 slot without direct pad candidate",
                ),
            )

    return missing_positions


def _balls_from_real_candidates(
    image: np.ndarray,
    candidates: list[_PadCandidate],
    pitch: float | None,
    config: HoughCircleConfig,
) -> list[SolderBall]:
    """Convert observed pad candidates into solder balls with refined radii."""
    circles: list[tuple[int, int, int, str]] = []
    for candidate in candidates:
        center_x = candidate.x
        center_y = candidate.y
        radius = candidate.radius
        if pitch is not None:
            seed_radius = max(
                radius,
                int(round(pitch * config.roi_pad_radius_seed_pitch_ratio)),
            )
            radius = _estimate_pad_radius(
                image=image,
                x=center_x,
                y=center_y,
                radius=seed_radius,
                config=config,
            )
            min_radius = int(round(pitch * config.roi_pad_radius_min_pitch_ratio))
            max_radius = int(round(pitch * config.roi_pad_radius_max_pitch_ratio))
            radius = max(min_radius, min(radius, max_radius))
            if config.use_local_component_final_snap:
                center_x, center_y, radius = _snap_pad_center_to_local_component(
                    image=image,
                    x=center_x,
                    y=center_y,
                    radius=radius,
                    pitch=pitch,
                    config=config,
                )
                center_x, center_y, radius = _snap_pad_center_to_local_hough(
                    image=image,
                    x=center_x,
                    y=center_y,
                    radius=radius,
                    pitch=pitch,
                    config=config,
                )
        circles.append((center_x, center_y, radius, candidate.source))

    expected_slots = config.expected_grid_rows * config.expected_grid_cols
    deduped_circles = _remove_duplicate_circles(
        [(x, y, radius) for x, y, radius, _source in circles],
        config,
        pitch=pitch,
    )
    source_by_circle = {
        (x, y, radius): source
        for x, y, radius, source in circles
    }
    circles = [
        (x, y, radius, source_by_circle.get((x, y, radius), "unknown"))
        for x, y, radius in deduped_circles
    ]
    if pitch is not None and len(circles) > expected_slots:
        circles = circles[:expected_slots]
    circles.sort(key=lambda item: (item[1], item[0]))
    return [
        SolderBall(
            ball_id=index + 1,
            center_x=x,
            center_y=y,
            radius=radius,
            confidence=0.82
            if source == "grid-evidence"
            else 1.0,
            is_estimated=source == "grid-evidence",
        )
        for index, (x, y, radius, source) in enumerate(circles)
    ]


def _snap_pad_center_to_local_component(
    image: np.ndarray,
    x: int,
    y: int,
    radius: int,
    pitch: float,
    config: HoughCircleConfig,
) -> tuple[int, int, int]:
    """Move a grid slot onto the closest dark pad component in its local crop."""
    height, width = image.shape[:2]
    min_radius = int(round(pitch * config.roi_pad_radius_min_pitch_ratio))
    max_radius = int(round(pitch * config.roi_pad_radius_max_pitch_ratio))
    max_offset = max(10.0, pitch * 0.62)
    search_radius = int(round(max(radius * 1.9, pitch * 0.68)))
    x0 = max(0, x - search_radius)
    y0 = max(0, y - search_radius)
    x1 = min(width - 1, x + search_radius)
    y1 = min(height - 1, y + search_radius)
    if x1 <= x0 or y1 <= y0:
        return x, y, radius

    crop = image[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        return x, y, radius

    blurred = cv2.GaussianBlur(crop, (5, 5), 0)
    otsu_threshold, otsu_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    fixed_threshold = min(160.0, max(90.0, float(otsu_threshold) + 8.0))
    fixed_mask = (blurred <= fixed_threshold).astype(np.uint8) * 255
    binary = cv2.bitwise_or(otsu_mask, fixed_mask)

    kernel_size = max(3, int(round(radius * 0.16)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return x, y, radius

    expected_area = pi * (radius**2)
    min_area = expected_area * 0.16
    # Parallax in oblique views doubles a pad with the shadow of the pad on
    # the other board side, so the merged dark blob can reach ~2x the pad
    # area; such blobs are still valid snap targets (handled below by using
    # the dark-core centroid instead of the full-blob centroid).
    max_area = expected_area * 2.30
    original_score = _grid_slot_evidence_score(image, x, y, radius)
    best: tuple[float, int, int, int, float] | None = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        box_x, box_y, box_width, box_height = cv2.boundingRect(contour)
        if box_width <= 0 or box_height <= 0:
            continue

        aspect_ratio = max(box_width, box_height) / max(1, min(box_width, box_height))
        if aspect_ratio > 3.60:
            continue

        extent = area / max(1.0, float(box_width * box_height))
        if extent < 0.24:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        circularity = (4.0 * pi * area) / (perimeter**2) if perimeter else 0.0
        if aspect_ratio > 2.35 and circularity < 0.12:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        center_x = int(round(x0 + (moments["m10"] / moments["m00"])))
        center_y = int(round(y0 + (moments["m01"] / moments["m00"])))
        effective_area = area
        if area > expected_area * 1.30:
            # Oversized blob = pad merged with a parallax shadow. Recenter on
            # the darkest core inside the blob (the actual solder pad) rather
            # than the centroid of the merged shape.
            contour_mask = np.zeros(crop.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
            core_threshold = max(40.0, float(otsu_threshold) - 12.0)
            core_mask = (blurred <= core_threshold) & (contour_mask > 0)
            core_ys, core_xs = np.nonzero(core_mask)
            if core_ys.size >= max(9.0, min_area * 0.5):
                center_x = int(round(x0 + float(np.mean(core_xs))))
                center_y = int(round(y0 + float(np.mean(core_ys))))
                effective_area = float(
                    min(core_ys.size, expected_area),
                )
        offset = float(np.hypot(center_x - x, center_y - y))
        if offset > max_offset:
            continue

        candidate_radius = int(round(np.sqrt(effective_area / pi)))
        candidate_radius = max(min_radius, min(candidate_radius, max_radius))
        evidence_score = _grid_slot_evidence_score(
            image=image,
            x=center_x,
            y=center_y,
            radius=max(candidate_radius, min_radius),
        )
        # Never trade evidence away: a snap that lands on a weaker dark
        # component than the current position corrupts the void measurement
        # (bright board area inside the circle reads as a false void).
        if evidence_score < max(0.34, original_score - 0.02):
            continue

        metrics = _dark_component_geometry_metrics(
            image=image,
            x=center_x,
            y=center_y,
            radius=max(candidate_radius, min_radius),
            config=config,
            intensity_threshold=158,
        )
        if metrics is None:
            continue

        component_aspect, component_area_ratio, component_circularity = metrics
        if component_area_ratio < 0.10:
            continue
        if component_aspect > 3.80:
            continue
        if component_aspect > 2.45 and component_circularity < 0.10:
            continue

        offset_score = 1.0 - min(1.0, offset / max_offset)
        shape_score = float(np.clip((extent + circularity) / 2.0, 0.0, 1.0))
        area_score = 1.0 - min(1.0, abs((area / expected_area) - 0.62) / 0.62)
        score = (
            (evidence_score * 1.7)
            + (offset_score * 0.7)
            + (shape_score * 0.5)
            + (area_score * 0.4)
        )
        if best is None or score > best[0]:
            best = (score, center_x, center_y, candidate_radius, evidence_score)

    if best is None:
        return x, y, radius

    score, center_x, center_y, candidate_radius, evidence_score = best
    moved = float(np.hypot(center_x - x, center_y - y)) > radius * 0.12
    stronger = evidence_score >= original_score + 0.03
    if moved or stronger or score >= original_score + 0.35:
        return center_x, center_y, candidate_radius

    return x, y, radius


def _snap_pad_center_to_local_hough(
    image: np.ndarray,
    x: int,
    y: int,
    radius: int,
    pitch: float,
    config: HoughCircleConfig,
) -> tuple[int, int, int]:
    """Move a final grid slot onto the nearest strong local pad circle."""
    height, width = image.shape[:2]
    max_offset = max(8.0, pitch * 0.42)
    search_radius = int(round(max(radius * 1.7, pitch * 0.45)))
    x0 = max(0, x - search_radius)
    y0 = max(0, y - search_radius)
    x1 = min(width - 1, x + search_radius)
    y1 = min(height - 1, y + search_radius)
    if x1 <= x0 or y1 <= y0:
        return x, y, radius

    crop = image[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        return x, y, radius

    zoom = max(1.0, min(2.4, 48.0 / max(1.0, radius)))
    search_crop = crop
    if zoom > 1.01:
        search_crop = cv2.resize(
            crop,
            None,
            fx=zoom,
            fy=zoom,
            interpolation=cv2.INTER_CUBIC,
        )

    blurred = cv2.GaussianBlur(search_crop, (7, 7), 1.2)
    min_zoom_radius = max(3, int(round(radius * zoom * 0.62)))
    max_zoom_radius = max(min_zoom_radius + 1, int(round(radius * zoom * 1.38)))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=max(8.0, radius * zoom * 0.80),
        param1=105.0,
        param2=10.0,
        minRadius=min_zoom_radius,
        maxRadius=max_zoom_radius,
    )
    if circles is None:
        return x, y, radius

    original_score = _grid_slot_evidence_score(image, x, y, radius)
    best: tuple[float, int, int, int, float] | None = None
    for local_x, local_y, local_radius in np.round(circles[0, :]).astype(int):
        center_x = int(round(x0 + (float(local_x) / zoom)))
        center_y = int(round(y0 + (float(local_y) / zoom)))
        offset = float(np.hypot(center_x - x, center_y - y))
        if offset > max_offset:
            continue

        candidate_radius = int(round(float(local_radius) / zoom))
        min_radius = int(round(pitch * config.roi_pad_radius_min_pitch_ratio))
        max_radius = int(round(pitch * config.roi_pad_radius_max_pitch_ratio))
        candidate_radius = max(min_radius, min(candidate_radius, max_radius))
        evidence_score = _grid_slot_evidence_score(
            image=image,
            x=center_x,
            y=center_y,
            radius=candidate_radius,
        )
        if evidence_score < 0.55:
            continue

        score = evidence_score - (0.22 * min(1.0, offset / max_offset))
        if best is None or score > best[0]:
            best = (score, center_x, center_y, candidate_radius, evidence_score)

    if best is None:
        return x, y, radius

    score, center_x, center_y, candidate_radius, evidence_score = best
    moved = float(np.hypot(center_x - x, center_y - y)) > radius * 0.18
    stronger = evidence_score >= original_score + 0.04
    safe_score = score >= max(0.40, original_score - 0.16)
    if safe_score and (moved or stronger):
        return center_x, center_y, candidate_radius

    return x, y, radius


def _estimate_pitch_from_candidates(
    candidates: list[_PadCandidate],
    config: HoughCircleConfig,
) -> float | None:
    """Estimate pitch from final real pad candidates."""
    if len(candidates) < 8:
        return None

    circles = [candidate.circle for candidate in candidates]
    pitch = _estimate_grid_pitch(circles, config)
    if pitch is not None:
        return pitch

    return None


def _roi_from_candidate_bounds(
    image: np.ndarray,
    candidates: list[tuple[int, int, int]],
    margin: float,
) -> RoiBounds | None:
    """Return a simple padded ROI around candidate circles."""
    if not candidates:
        return None

    x_values = [x for x, _, _ in candidates]
    y_values = [y for _, y, _ in candidates]
    radii = [radius for _, _, radius in candidates]
    pad = margin if margin > 0.0 else float(np.median(radii) if radii else 0.0)
    return _clamp_roi(
        RoiBounds(
            x_min=int(round(min(x_values) - pad)),
            y_min=int(round(min(y_values) - pad)),
            x_max=int(round(max(x_values) + pad)),
            y_max=int(round(max(y_values) + pad)),
        ),
        image.shape[:2],
    )


def _clamp_roi(roi: RoiBounds, image_shape: tuple[int, int]) -> RoiBounds:
    """Clamp ROI bounds to image dimensions."""
    height, width = image_shape
    return RoiBounds(
        x_min=max(0, min(width - 1, roi.x_min)),
        y_min=max(0, min(height - 1, roi.y_min)),
        x_max=max(0, min(width, roi.x_max)),
        y_max=max(0, min(height, roi.y_max)),
    )


def _roi_contains_point(roi: RoiBounds, x: int, y: int) -> bool:
    """Return true when a point is inside an ROI."""
    return roi.x_min <= x <= roi.x_max and roi.y_min <= y <= roi.y_max


def _rejected_real_candidates(
    candidates: list[_PadCandidate],
    kept_candidates: set[tuple[int, int, int]],
    roi: RoiBounds | None,
) -> list[tuple[int, int, int]]:
    """Return raw candidates not kept by the final real-pad filter."""
    rejected: list[tuple[int, int, int]] = []
    for candidate in candidates:
        if candidate.circle in kept_candidates:
            continue
        if roi is None or _roi_contains_point(roi, candidate.x, candidate.y):
            rejected.append(candidate.circle)
        else:
            rejected.append(candidate.circle)

    return rejected


def _detect_bga_roi_candidates(
    image: np.ndarray,
    config: HoughCircleConfig,
) -> list[tuple[int, int, int]]:
    """Detect broad circle candidates for locating the dense BGA grid ROI."""
    height, width = image.shape[:2]
    scale = min(1.0, config.roi_detection_max_dimension / max(height, width))
    if scale < 1.0:
        small = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image

    circles = cv2.HoughCircles(
        small,
        cv2.HOUGH_GRADIENT,
        dp=config.dp,
        minDist=config.roi_candidate_min_dist_scaled,
        param1=config.param1,
        param2=config.roi_candidate_param2,
        minRadius=config.roi_candidate_min_radius_scaled,
        maxRadius=config.roi_candidate_max_radius_scaled,
    )
    if circles is None:
        return []

    candidates: list[tuple[int, int, int]] = []
    for x, y, radius in np.round(circles[0]).astype(int):
        if (
            x - radius < 0
            or y - radius < 0
            or x + radius >= small.shape[1]
            or y + radius >= small.shape[0]
        ):
            continue
        if not _has_scaled_candidate_evidence(
            image=small,
            x=int(x),
            y=int(y),
            radius=int(radius),
            config=config,
        ):
            continue

        candidates.append(
            (
                int(round(x / scale)),
                int(round(y / scale)),
                max(1, int(round(radius / scale))),
            ),
        )

    return _remove_duplicate_circles(candidates, config)


def _has_scaled_candidate_evidence(
    image: np.ndarray,
    x: int,
    y: int,
    radius: int,
    config: HoughCircleConfig,
) -> bool:
    """Check basic intensity and geometry for coarse ROI circle candidates."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 255, thickness=-1)
    pixels = image[mask > 0]
    if pixels.size == 0:
        return False

    mean_intensity = float(np.mean(pixels))
    dark_fraction = float(
        np.count_nonzero(pixels <= config.roi_candidate_dark_threshold),
    ) / float(pixels.size)
    if mean_intensity > config.roi_candidate_max_mean_intensity:
        return False
    if dark_fraction < config.roi_candidate_min_dark_fraction:
        return False

    metrics = _dark_component_geometry_metrics(
        image=image,
        x=x,
        y=y,
        radius=radius,
        config=config,
        intensity_threshold=config.roi_candidate_dark_threshold,
    )
    if metrics is None:
        return True

    aspect_ratio, area_ratio, circularity = metrics
    return not (aspect_ratio > 4.0 and area_ratio < 0.20 and circularity < 0.18)


def _fit_bga_grid(
    candidates: list[tuple[int, int, int]],
    config: HoughCircleConfig,
) -> _BgaGridFit | None:
    """Fit the most coherent expected BGA grid over raw circle candidates."""
    expected_slots = config.expected_grid_rows * config.expected_grid_cols
    if len(candidates) < max(30, expected_slots // 4):
        return None

    points = np.array([(x, y) for x, y, _ in candidates], dtype=np.float32)
    pitches = _bga_pitch_candidates(candidates, config)
    if not pitches:
        return None

    best: _BgaGridFit | None = None
    for pitch in pitches:
        x_sets = _best_axis_position_sets(
            values=points[:, 0],
            pitch=pitch,
            line_count=config.expected_grid_cols,
            keep=config.grid_axis_candidate_keep_count,
            config=config,
        )
        y_sets = _best_axis_position_sets(
            values=points[:, 1],
            pitch=pitch,
            line_count=config.expected_grid_rows,
            keep=config.grid_axis_candidate_keep_count,
            config=config,
        )
        slot_tolerance = max(10.0, pitch * config.grid_slot_tolerance_ratio)
        for x_score, x_positions in x_sets:
            for y_score, y_positions in y_sets:
                occupied = _grid_occupancy(
                    points=points,
                    x_positions=x_positions,
                    y_positions=y_positions,
                    slot_tolerance=slot_tolerance,
                )
                area = (x_positions[-1] - x_positions[0]) * (
                    y_positions[-1] - y_positions[0]
                )
                score = (occupied * 100.0) + x_score + y_score - (area * 0.00002)
                if best is None or score > best.score:
                    best = _BgaGridFit(
                        score=score,
                        occupied=occupied,
                        pitch_x=pitch,
                        pitch_y=pitch,
                        x_positions=tuple(x_positions),
                        y_positions=tuple(y_positions),
                    )

    return best


def _bga_pitch_candidates(
    candidates: list[tuple[int, int, int]],
    config: HoughCircleConfig,
) -> list[float]:
    """Estimate likely BGA pitch values from same-row and same-column gaps."""
    points = np.array([(x, y) for x, y, _ in candidates], dtype=np.float32)
    if len(points) > 700:
        indexes = np.linspace(0, len(points) - 1, 700).astype(int)
        points = points[indexes]

    axis_tolerance = 18.0
    gaps: list[float] = []
    for x, y in points:
        dx = np.abs(points[:, 0] - x)
        dy = np.abs(points[:, 1] - y)
        row_gaps = dx[
            (dy <= axis_tolerance)
            & (dx >= config.grid_search_min_pitch)
            & (dx <= config.grid_search_max_pitch)
        ]
        column_gaps = dy[
            (dx <= axis_tolerance)
            & (dy >= config.grid_search_min_pitch)
            & (dy <= config.grid_search_max_pitch)
        ]
        gaps.extend(row_gaps.tolist())
        gaps.extend(column_gaps.tolist())

    if not gaps:
        return []

    bin_width = config.grid_pitch_histogram_bin
    bins = np.arange(
        config.grid_search_min_pitch,
        config.grid_search_max_pitch + bin_width,
        bin_width,
    )
    if bins.size < 2:
        return []

    histogram, edges = np.histogram(gaps, bins=bins)
    peak_indexes = np.argsort(histogram)[-config.grid_pitch_keep_count :][::-1]
    pitches: list[float] = []
    for index in peak_indexes:
        if histogram[index] < 5:
            continue
        pitch = float((edges[index] + edges[index + 1]) / 2.0)
        if all(abs(pitch - existing) > bin_width * 2 for existing in pitches):
            pitches.append(pitch)

    return pitches


def _best_axis_position_sets(
    values: np.ndarray,
    pitch: float,
    line_count: int,
    keep: int,
    config: HoughCircleConfig,
) -> list[tuple[float, tuple[float, ...]]]:
    """Return likely regular line sequences for one grid axis."""
    tolerance = max(10.0, pitch * config.grid_line_tolerance_ratio)
    starts: set[float] = set()
    for value in values:
        for index in range(line_count):
            start = float(value - (index * pitch))
            starts.add(round(start / 3.0) * 3.0)

    scored: list[tuple[float, tuple[float, ...]]] = []
    for start in starts:
        positions = tuple(start + (index * pitch) for index in range(line_count))
        position_array = np.array(positions, dtype=np.float32)
        distances = np.abs(values[:, None] - position_array[None, :])
        counts = np.count_nonzero(distances <= tolerance, axis=0)
        occupied_lines = int(np.count_nonzero(counts > 0))
        capped_count = int(np.sum(np.minimum(counts, line_count)))
        edge_penalty = 4 if counts[0] == 0 or counts[-1] == 0 else 0
        score = (occupied_lines * 10.0) + capped_count - edge_penalty
        scored.append((score, positions))

    scored.sort(key=lambda item: item[0], reverse=True)
    unique: list[tuple[float, tuple[float, ...]]] = []
    for score, positions in scored:
        if any(
            abs(positions[0] - existing_positions[0]) < pitch * 0.2
            for _, existing_positions in unique
        ):
            continue
        unique.append((score, positions))
        if len(unique) >= keep:
            break

    return unique


def _grid_occupancy(
    points: np.ndarray,
    x_positions: tuple[float, ...],
    y_positions: tuple[float, ...],
    slot_tolerance: float,
) -> int:
    """Count grid intersections that have at least one nearby candidate."""
    occupied = 0
    for y in y_positions:
        for x in x_positions:
            distances = np.hypot(points[:, 0] - x, points[:, 1] - y)
            if np.any(distances <= slot_tolerance):
                occupied += 1
    return occupied


def _refine_bga_grid_from_candidates(
    candidates: list[tuple[int, int, int]],
    grid: _BgaGridFit,
    config: HoughCircleConfig,
) -> _BgaGridFit:
    """Nudge fitted grid lines toward nearby candidate centers."""
    assignments = _assign_candidates_to_grid(candidates, grid, config)
    x_positions = list(grid.x_positions)
    y_positions = list(grid.y_positions)

    for column in range(config.expected_grid_cols):
        values = [
            float(candidates[index][0])
            for (row, assigned_column), index in assignments.items()
            if assigned_column == column
        ]
        if len(values) >= config.grid_refine_min_axis_assignments:
            x_positions[column] = float(np.median(values))

    for row in range(config.expected_grid_rows):
        values = [
            float(candidates[index][1])
            for (assigned_row, column), index in assignments.items()
            if assigned_row == row
        ]
        if len(values) >= config.grid_refine_min_axis_assignments:
            y_positions[row] = float(np.median(values))

    points = np.array([(x, y) for x, y, _ in candidates], dtype=np.float32)
    pitch_x = _median_axis_pitch(x_positions, grid.pitch_x)
    pitch_y = _median_axis_pitch(y_positions, grid.pitch_y)
    occupied = _grid_occupancy(
        points=points,
        x_positions=tuple(x_positions),
        y_positions=tuple(y_positions),
        slot_tolerance=max(10.0, min(pitch_x, pitch_y) * config.grid_slot_tolerance_ratio),
    )
    return _BgaGridFit(
        score=grid.score,
        occupied=occupied,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        x_positions=tuple(x_positions),
        y_positions=tuple(y_positions),
    )


def _median_axis_pitch(positions: list[float], fallback: float) -> float:
    """Return the median spacing between sorted grid lines."""
    gaps = np.diff(np.array(sorted(positions), dtype=np.float32))
    valid_gaps = gaps[gaps > 0]
    if valid_gaps.size == 0:
        return fallback
    return float(np.median(valid_gaps))


def _assign_candidates_to_grid(
    candidates: list[tuple[int, int, int]],
    grid: _BgaGridFit,
    config: HoughCircleConfig,
) -> dict[tuple[int, int], int]:
    """Assign at most one raw candidate to each expected grid slot."""
    if not candidates:
        return {}

    points = np.array([(x, y) for x, y, _ in candidates], dtype=np.float32)
    tolerance = max(
        10.0,
        min(grid.pitch_x, grid.pitch_y) * config.grid_slot_tolerance_ratio,
    )
    assignments: dict[tuple[int, int], int] = {}
    used: set[int] = set()
    for row, y in enumerate(grid.y_positions):
        for column, x in enumerate(grid.x_positions):
            distances = np.hypot(points[:, 0] - x, points[:, 1] - y)
            for candidate_index in np.argsort(distances):
                index = int(candidate_index)
                if index in used:
                    continue
                if distances[index] > tolerance:
                    break
                assignments[(row, column)] = index
                used.add(index)
                break

    return assignments


def _build_bga_grid_balls(
    image: np.ndarray,
    grid: _BgaGridFit,
    candidates: list[tuple[int, int, int]],
    config: HoughCircleConfig,
) -> tuple[list[SolderBall], list[MissingGridPosition], set[tuple[int, int, int]]]:
    """Build row-major solder balls from the fitted 16x16 BGA grid."""
    assignments = _assign_candidates_to_grid(candidates, grid, config)
    pitch = float((grid.pitch_x + grid.pitch_y) / 2.0)
    base_radius = max(config.grid_radius_min, int(round(pitch * config.grid_radius_pitch_ratio)))
    balls: list[SolderBall] = []
    missing_positions: list[MissingGridPosition] = []
    assigned_candidates: set[tuple[int, int, int]] = set()

    ball_id = 1
    for row, y in enumerate(grid.y_positions):
        for column, x in enumerate(grid.x_positions):
            center_x = int(round(x))
            center_y = int(round(y))
            assignment_index = assignments.get((row, column))
            confidence = 0.45
            is_estimated = assignment_index is None
            if assignment_index is not None:
                assigned_candidates.add(candidates[assignment_index])
                confidence = 1.0
            else:
                confidence = _grid_slot_evidence_score(
                    image=image,
                    x=center_x,
                    y=center_y,
                    radius=base_radius,
                )
                if confidence < 0.35:
                    missing_positions.append(
                        MissingGridPosition(
                            row=row + 1,
                            column=column + 1,
                            center_x=center_x,
                            center_y=center_y,
                            reason="no nearby circle candidate / weak local pad evidence",
                        ),
                    )

            radius = _estimate_pad_radius(
                image=image,
                x=center_x,
                y=center_y,
                radius=base_radius,
                config=config,
            )
            balls.append(
                SolderBall(
                    ball_id=ball_id,
                    center_x=center_x,
                    center_y=center_y,
                    radius=radius,
                    confidence=round(confidence, 3),
                    is_estimated=is_estimated,
                ),
            )
            ball_id += 1

    return balls, missing_positions, assigned_candidates


def _grid_slot_evidence_score(
    image: np.ndarray,
    x: int,
    y: int,
    radius: int,
) -> float:
    """Score local evidence that an expected grid slot contains a dark pad."""
    height, width = image.shape[:2]
    if x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height:
        return 0.0

    crop = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
    yy, xx = np.indices(crop.shape)
    distances = np.hypot(xx - radius, yy - radius)
    circle_mask = distances <= radius
    annulus_mask = (distances <= radius * 0.96) & (distances >= radius * 0.52)
    circle_pixels = crop[circle_mask]
    annulus_pixels = crop[annulus_mask]
    if circle_pixels.size == 0 or annulus_pixels.size == 0:
        return 0.0

    circle_mean = float(np.mean(circle_pixels))
    annulus_mean = float(np.mean(annulus_pixels))
    dark_fraction = float(np.count_nonzero(circle_pixels <= 150)) / float(
        circle_pixels.size,
    )
    annulus_dark_fraction = float(
        np.count_nonzero(annulus_pixels <= 150),
    ) / float(annulus_pixels.size)

    intensity_score = np.clip((190.0 - circle_mean) / 80.0, 0.0, 1.0)
    annulus_score = np.clip((185.0 - annulus_mean) / 85.0, 0.0, 1.0)
    dark_score = np.clip(dark_fraction / 0.45, 0.0, 1.0)
    ring_score = np.clip(annulus_dark_fraction / 0.45, 0.0, 1.0)
    pad_score = float(
        (intensity_score + annulus_score + dark_score + ring_score) / 4.0,
    )
    void_score = _bright_void_evidence_score(image, x, y, radius)
    return float(max(pad_score, (pad_score * 0.85) + (void_score * 0.15)))


def _bright_void_evidence_score(
    image: np.ndarray,
    x: int,
    y: int,
    radius: int,
) -> float:
    """Score bright compact void-like spots inside a dark pad candidate."""
    height, width = image.shape[:2]
    if x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height:
        return 0.0

    crop = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
    yy, xx = np.indices(crop.shape)
    distances = np.hypot(xx - radius, yy - radius)
    circle_mask = distances <= radius * 0.86
    pixels = crop[circle_mask]
    if pixels.size == 0:
        return 0.0

    local_mean = float(np.mean(pixels))
    local_std = float(np.std(pixels))
    threshold = max(
        float(np.percentile(pixels, 82)),
        local_mean + max(18.0, local_std * 0.55),
    )
    bright_mask = ((crop >= threshold) & circle_mask).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
    pad_area = float(pi * (radius**2))
    accepted_area = 0.0
    accepted_count = 0
    for label in range(1, stats.shape[0]):
        area = float(stats[label, cv2.CC_STAT_AREA])
        area_ratio = area / max(1.0, pad_area)
        if (
            area_ratio < HoughCircleConfig.void_evidence_min_area_ratio
            or area_ratio > HoughCircleConfig.void_evidence_max_area_ratio
        ):
            continue

        box_width = float(stats[label, cv2.CC_STAT_WIDTH])
        box_height = float(stats[label, cv2.CC_STAT_HEIGHT])
        aspect_ratio = max(box_width, box_height) / max(1.0, min(box_width, box_height))
        if aspect_ratio > 2.4:
            continue

        accepted_area += area
        accepted_count += 1

    if accepted_count == 0:
        return 0.0

    area_score = np.clip(accepted_area / max(1.0, pad_area * 0.08), 0.0, 1.0)
    count_score = np.clip(accepted_count / 4.0, 0.0, 1.0)
    contrast_score = np.clip(
        (threshold - local_mean) / HoughCircleConfig.void_evidence_min_intensity_delta,
        0.0,
        1.0,
    )
    return float((area_score + count_score + contrast_score) / 3.0)


def _grid_roi_bounds(
    image: np.ndarray,
    grid: _BgaGridFit,
    config: HoughCircleConfig,
) -> RoiBounds:
    """Return a padded ROI around the fitted BGA grid."""
    height, width = image.shape[:2]
    margin_x = grid.pitch_x * config.grid_roi_margin_pitch_ratio
    margin_y = grid.pitch_y * config.grid_roi_margin_pitch_ratio
    return RoiBounds(
        x_min=max(0, int(round(grid.x_positions[0] - margin_x))),
        y_min=max(0, int(round(grid.y_positions[0] - margin_y))),
        x_max=min(width, int(round(grid.x_positions[-1] + margin_x))),
        y_max=min(height, int(round(grid.y_positions[-1] + margin_y))),
    )


def _rejected_grid_candidates(
    candidates: list[tuple[int, int, int]],
    assigned_candidates: set[tuple[int, int, int]],
    roi: RoiBounds,
) -> list[tuple[int, int, int]]:
    """Return raw candidates not used by the final fitted BGA grid."""
    rejected = []
    for candidate in candidates:
        x, y, _ = candidate
        if candidate in assigned_candidates:
            continue
        if roi.x_min <= x <= roi.x_max and roi.y_min <= y <= roi.y_max:
            rejected.append(candidate)
        else:
            rejected.append(candidate)
    return rejected


def _is_dark_circle(
    image: np.ndarray,
    x: int,
    y: int,
    radius: int,
    config: HoughCircleConfig,
) -> bool:
    """Keep circles whose interior is dark enough to be solder balls."""
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    if config.require_complete_circle:
        if (
            x - radius < 0
            or y - radius < 0
            or x + radius >= width
            or y + radius >= height
        ):
            return False

    circle_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(circle_mask, (x, y), radius, 255, thickness=-1)
    mean_intensity = cv2.mean(image, mask=circle_mask)[0]
    if mean_intensity > config.max_circle_mean_intensity:
        return False

    core_radius = max(1, int(radius * config.core_radius_scale))
    core_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(core_mask, (x, y), core_radius, 255, thickness=-1)
    core_mean_intensity = cv2.mean(image, mask=core_mask)[0]
    if core_mean_intensity > config.max_core_mean_intensity:
        return False

    circle_pixels = image[circle_mask > 0]
    dark_fraction = float(
        np.count_nonzero(
            circle_pixels <= config.dark_pixel_intensity_threshold,
        )
    ) / float(circle_pixels.size)
    if dark_fraction < config.min_dark_pixel_fraction:
        return False

    inner_ring_radius = max(1, int(radius * config.outer_ring_inner_radius_scale))
    inner_ring_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(inner_ring_mask, (x, y), inner_ring_radius, 255, thickness=-1)
    outer_ring_mask = cv2.bitwise_and(circle_mask, cv2.bitwise_not(inner_ring_mask))
    outer_ring_pixels = image[outer_ring_mask > 0]
    outer_dark_fraction = float(
        np.count_nonzero(
            outer_ring_pixels <= config.dark_pixel_intensity_threshold,
        )
    ) / float(outer_ring_pixels.size)
    if outer_dark_fraction < config.min_outer_dark_pixel_fraction:
        return False

    return _has_coherent_dark_pad_component(image, x, y, radius, config)


def _has_coherent_dark_pad_component(
    image: np.ndarray,
    x: int,
    y: int,
    radius: int,
    config: HoughCircleConfig,
) -> bool:
    """Reject line-like dark structures that accidentally form Hough circles."""
    if not config.reject_trace_like_components:
        return True

    height, width = image.shape[:2]
    x_min = max(x - radius, 0)
    y_min = max(y - radius, 0)
    x_max = min(x + radius + 1, width)
    y_max = min(y + radius + 1, height)
    crop = image[y_min:y_max, x_min:x_max]
    local_x = x - x_min
    local_y = y - y_min

    yy, xx = np.indices(crop.shape)
    distances = np.hypot(xx - local_x, yy - local_y)
    circle_mask = distances <= radius
    dark_mask = ((crop <= config.dark_pixel_intensity_threshold) & circle_mask).astype(
        np.uint8,
    )
    kernel_size = max(1, config.dark_component_close_kernel_size)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    seed_radius = max(3, int(radius * config.dark_component_seed_radius_scale))
    seed_mask = distances <= seed_radius
    seed_labels = [label for label in np.unique(labels[seed_mask]) if label != 0]
    if not seed_labels:
        return True

    component_label = max(
        seed_labels,
        key=lambda label: stats[label, cv2.CC_STAT_AREA],
    )
    component_area = float(stats[component_label, cv2.CC_STAT_AREA])
    component_width = float(stats[component_label, cv2.CC_STAT_WIDTH])
    component_height = float(stats[component_label, cv2.CC_STAT_HEIGHT])
    component_aspect_ratio = max(component_width, component_height) / max(
        1.0,
        min(component_width, component_height),
    )
    component_area_ratio = component_area / (pi * (radius**2))

    is_trace_like = (
        component_aspect_ratio > config.max_trace_component_aspect_ratio
        and component_area_ratio < config.max_trace_component_area_ratio
    )
    return not is_trace_like


def _dark_component_shape_metrics(
    image: np.ndarray,
    x: int,
    y: int,
    radius: int,
    config: HoughCircleConfig,
) -> tuple[float, float] | None:
    """Return aspect ratio and area ratio for the central dark component."""
    metrics = _dark_component_geometry_metrics(
        image=image,
        x=x,
        y=y,
        radius=radius,
        config=config,
    )
    if metrics is None:
        return None

    aspect_ratio, area_ratio, _ = metrics
    return aspect_ratio, area_ratio


def _dark_component_geometry_metrics(
    image: np.ndarray,
    x: int,
    y: int,
    radius: int,
    config: HoughCircleConfig,
    intensity_threshold: int | None = None,
) -> tuple[float, float, float] | None:
    """Return aspect ratio, area ratio, and circularity for the central component."""
    height, width = image.shape[:2]
    x_min = max(x - radius, 0)
    y_min = max(y - radius, 0)
    x_max = min(x + radius + 1, width)
    y_max = min(y + radius + 1, height)
    crop = image[y_min:y_max, x_min:x_max]
    local_x = x - x_min
    local_y = y - y_min

    yy, xx = np.indices(crop.shape)
    distances = np.hypot(xx - local_x, yy - local_y)
    circle_mask = distances <= radius
    threshold = (
        config.dark_pixel_intensity_threshold
        if intensity_threshold is None
        else intensity_threshold
    )
    dark_mask = ((crop <= threshold) & circle_mask).astype(
        np.uint8,
    )
    kernel_size = max(1, config.dark_component_close_kernel_size)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    seed_radius = max(3, int(radius * config.dark_component_seed_radius_scale))
    seed_mask = distances <= seed_radius
    seed_labels = [label for label in np.unique(labels[seed_mask]) if label != 0]
    if not seed_labels:
        return None

    component_label = max(
        seed_labels,
        key=lambda label: stats[label, cv2.CC_STAT_AREA],
    )
    component_area = float(stats[component_label, cv2.CC_STAT_AREA])
    component_width = float(stats[component_label, cv2.CC_STAT_WIDTH])
    component_height = float(stats[component_label, cv2.CC_STAT_HEIGHT])
    component_aspect_ratio = max(component_width, component_height) / max(
        1.0,
        min(component_width, component_height),
    )
    component_area_ratio = component_area / (pi * (radius**2))
    component_mask = (labels == component_label).astype(np.uint8)
    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        circularity = 0.0
    else:
        perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), True))
        circularity = (
            (4.0 * pi * component_area) / (perimeter**2)
            if perimeter > 0.0
            else 0.0
        )

    return component_aspect_ratio, component_area_ratio, float(circularity)


def _remove_duplicate_circles(
    circles: list[tuple[int, int, int]],
    config: HoughCircleConfig,
    pitch: float | None = None,
) -> list[tuple[int, int, int]]:
    """Remove overlapping repeated Hough detections."""
    kept: list[tuple[int, int, int]] = []
    for circle in sorted(circles, key=lambda item: item[2], reverse=True):
        x, y, radius = circle
        duplicate = False
        for kept_x, kept_y, kept_radius in kept:
            distance = np.hypot(x - kept_x, y - kept_y)
            threshold = min(radius, kept_radius) * config.duplicate_distance_factor
            if pitch is not None:
                threshold = max(threshold, pitch * 0.45)
            if distance < threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(circle)
    return kept


def _remove_layout_outliers(
    circles: list[tuple[int, int, int]],
    config: HoughCircleConfig,
) -> list[tuple[int, int, int]]:
    """Remove isolated via-like circles that do not fit the BGA grid layout."""
    if len(circles) < 3:
        return circles

    kept: list[tuple[int, int, int]] = []
    for index, (x, y, radius) in enumerate(circles):
        aligned_neighbors = 0
        for other_index, (other_x, other_y, _) in enumerate(circles):
            if index == other_index:
                continue

            same_row = abs(y - other_y) <= config.alignment_tolerance
            row_pitch = abs(x - other_x)
            same_column = abs(x - other_x) <= config.alignment_tolerance
            column_pitch = abs(y - other_y)
            has_row_neighbor = (
                same_row
                and config.min_neighbor_pitch <= row_pitch <= config.max_neighbor_pitch
            )
            has_column_neighbor = (
                same_column
                and config.min_neighbor_pitch <= column_pitch <= config.max_neighbor_pitch
            )
            if has_row_neighbor or has_column_neighbor:
                aligned_neighbors += 1

        if aligned_neighbors >= config.min_aligned_neighbors:
            kept.append((x, y, radius))

    return kept


def _normalize_small_pad_radii(
    circles: list[tuple[int, int, int]],
    config: HoughCircleConfig,
) -> list[tuple[int, int, int]]:
    """Lift inner-only Hough radii toward the image-level pad radius."""
    if len(circles) < config.normalized_radius_min_count:
        return circles

    median_radius = float(np.median([radius for _, _, radius in circles]))
    radius_floor = int(round(median_radius * config.normalized_radius_median_ratio))
    normalized: list[tuple[int, int, int]] = []
    for x, y, radius in circles:
        capped_floor = min(radius + config.normalized_radius_max_boost, radius_floor)
        normalized.append((x, y, max(radius, capped_floor)))

    return normalized


def _recover_grid_gap_circles(
    image: np.ndarray,
    circles: list[tuple[int, int, int]],
    config: HoughCircleConfig,
) -> list[tuple[int, int, int]]:
    """Recover dark pad candidates that sit in clear row or column grid gaps."""
    if len(circles) < config.grid_gap_min_count:
        return circles

    pitch = _estimate_grid_pitch(circles, config)
    if pitch is None:
        return circles

    recovery_radius = int(round(np.median([radius for _, _, radius in circles])))
    recovered = list(circles)
    for axis in (0, 1):
        groups = _cluster_by_axis(circles, axis=axis, config=config)
        for group in groups:
            if len(group) < 3:
                continue

            sorted_group = sorted(group, key=lambda item: item[1 - axis])
            for left, right in zip(sorted_group, sorted_group[1:]):
                gap = right[1 - axis] - left[1 - axis]
                missing_count = int(round(gap / pitch)) - 1
                if missing_count < 1:
                    continue

                expected_gap = pitch * (missing_count + 1)
                gap_error = abs(gap - expected_gap)
                if gap_error > pitch * config.grid_gap_pitch_tolerance_ratio:
                    continue

                for step in range(1, missing_count + 1):
                    fraction = step / (missing_count + 1)
                    candidate_x = int(round(left[0] + ((right[0] - left[0]) * fraction)))
                    candidate_y = int(round(left[1] + ((right[1] - left[1]) * fraction)))
                    candidate = (candidate_x, candidate_y, recovery_radius)
                    if _is_near_existing_circle(candidate, recovered, config):
                        continue
                    if _is_grid_gap_pad_candidate(image, candidate, config):
                        recovered.append(candidate)

    return _remove_duplicate_circles(recovered, config)


def _regularize_grid_centers(
    circles: list[tuple[int, int, int]],
    config: HoughCircleConfig,
) -> list[tuple[int, int, int]]:
    """Snap BGA centers to stable row/column grid-line intersections."""
    x_positions = _regularized_axis_positions(circles, axis=0, config=config)
    y_positions = _regularized_axis_positions(circles, axis=1, config=config)
    if x_positions is None or y_positions is None:
        return circles

    grid_slot_count = len(x_positions) * len(y_positions)
    fill_ratio = len(circles) / float(grid_slot_count)
    if fill_ratio < config.grid_regularization_min_fill_ratio:
        return circles

    regularized: list[tuple[int, int, int]] = []
    for x, y, radius in circles:
        regularized_x = _nearest_grid_position(
            value=x,
            positions=x_positions,
            tolerance=config.grid_regularization_snap_tolerance,
        )
        regularized_y = _nearest_grid_position(
            value=y,
            positions=y_positions,
            tolerance=config.grid_regularization_snap_tolerance,
        )
        if regularized_x is None or regularized_y is None:
            regularized.append((x, y, radius))
            continue

        regularized.append((regularized_x, regularized_y, radius))

    return _remove_duplicate_circles(regularized, config)


def _regularized_axis_positions(
    circles: list[tuple[int, int, int]],
    axis: int,
    config: HoughCircleConfig,
) -> list[float] | None:
    """Return stable grid-line positions for one image axis."""
    clusters = _cluster_by_axis(circles, axis=axis, config=config)
    if len(clusters) < config.grid_regularization_min_axis_count:
        return None

    positions = [float(np.median([item[axis] for item in cluster])) for cluster in clusters]
    cluster_stds = [float(np.std([item[axis] for item in cluster])) for cluster in clusters]
    if max(cluster_stds) > config.grid_regularization_max_cluster_std:
        return None

    sorted_positions = sorted(positions)
    gaps = np.diff(np.array(sorted_positions, dtype=np.float32))
    valid_gaps = gaps[
        (gaps >= config.min_neighbor_pitch)
        & (gaps <= config.max_neighbor_pitch)
    ]
    if valid_gaps.size < max(1, len(sorted_positions) - 1):
        return None

    median_gap = float(np.median(valid_gaps))
    if median_gap <= 0:
        return None

    pitch_cv = float(np.std(valid_gaps) / median_gap)
    if pitch_cv > config.grid_regularization_max_pitch_cv:
        return None

    return sorted_positions


def _nearest_grid_position(
    value: int,
    positions: list[float],
    tolerance: float,
) -> int | None:
    """Return the nearest regularized grid position when it is close enough."""
    nearest = min(positions, key=lambda position: abs(position - value))
    if abs(nearest - value) > tolerance:
        return None

    return int(round(nearest))


def _normalize_contour_radii(
    circles: list[tuple[int, int, int]],
    config: HoughCircleConfig,
) -> list[tuple[int, int, int]]:
    """Lift conservative contour radii toward the robust image-level pad radius."""
    if len(circles) < config.contour_radius_min_count:
        return circles

    radius_floor = float(
        np.percentile(
            [radius for _, _, radius in circles],
            config.contour_radius_floor_percentile,
        ),
    )
    pitch = _estimate_grid_pitch(circles, config)
    if pitch is not None:
        radius_floor = min(radius_floor, pitch * config.contour_radius_max_pitch_ratio)

    normalized: list[tuple[int, int, int]] = []
    for x, y, radius in circles:
        boosted_radius = min(
            int(round(radius_floor)),
            radius + config.contour_radius_max_boost,
        )
        normalized.append((x, y, max(radius, boosted_radius)))

    return normalized


def _estimate_grid_pitch(
    circles: list[tuple[int, int, int]],
    config: HoughCircleConfig,
) -> float | None:
    """Estimate the dominant BGA pitch from adjacent row and column gaps."""
    gaps: list[int] = []
    for axis in (0, 1):
        groups = _cluster_by_axis(circles, axis=axis, config=config)
        for group in groups:
            if len(group) < 3:
                continue

            sorted_positions = sorted(item[1 - axis] for item in group)
            for left, right in zip(sorted_positions, sorted_positions[1:]):
                gap = right - left
                if config.min_neighbor_pitch <= gap <= config.max_neighbor_pitch:
                    gaps.append(gap)

    if not gaps:
        return None

    return float(np.median(gaps))


def _cluster_by_axis(
    circles: list[tuple[int, int, int]],
    axis: int,
    config: HoughCircleConfig,
) -> list[list[tuple[int, int, int]]]:
    """Cluster circles into approximate rows or columns."""
    clusters: list[list[tuple[int, int, int]]] = []
    sorted_circles = sorted(circles, key=lambda item: item[axis])
    for circle in sorted_circles:
        coordinate = circle[axis]
        for cluster in clusters:
            cluster_coordinate = float(np.mean([item[axis] for item in cluster]))
            if abs(coordinate - cluster_coordinate) <= config.alignment_tolerance:
                cluster.append(circle)
                break
        else:
            clusters.append([circle])

    return clusters


def _is_near_existing_circle(
    candidate: tuple[int, int, int],
    circles: list[tuple[int, int, int]],
    config: HoughCircleConfig,
) -> bool:
    """Return true when a grid-gap candidate duplicates an existing circle."""
    x, y, _ = candidate
    return any(
        np.hypot(x - existing_x, y - existing_y) <= config.grid_gap_duplicate_distance
        for existing_x, existing_y, _ in circles
    )


def _is_grid_gap_pad_candidate(
    image: np.ndarray,
    candidate: tuple[int, int, int],
    config: HoughCircleConfig,
) -> bool:
    """Validate a recovered grid-gap candidate with local pad appearance."""
    x, y, radius = candidate
    height, width = image.shape[:2]
    if (
        x - radius < 0
        or y - radius < 0
        or x + radius >= width
        or y + radius >= height
    ):
        return False

    x_min = x - radius
    y_min = y - radius
    x_max = x + radius + 1
    y_max = y + radius + 1
    crop = image[y_min:y_max, x_min:x_max]
    yy, xx = np.indices(crop.shape)
    distances = np.hypot(xx - radius, yy - radius)
    circle_mask = distances <= radius
    core_mask = distances <= radius * config.core_radius_scale
    outer_mask = (distances <= radius) & (
        distances >= radius * config.outer_ring_inner_radius_scale
    )

    circle_pixels = crop[circle_mask]
    core_pixels = crop[core_mask]
    outer_pixels = crop[outer_mask]
    mean_intensity = float(np.mean(circle_pixels))
    core_mean_intensity = float(np.mean(core_pixels))
    dark_fraction = float(
        np.count_nonzero(circle_pixels <= config.dark_pixel_intensity_threshold),
    ) / float(circle_pixels.size)
    outer_dark_fraction = float(
        np.count_nonzero(outer_pixels <= config.dark_pixel_intensity_threshold),
    ) / float(outer_pixels.size)

    if mean_intensity > config.grid_gap_max_mean_intensity:
        return False
    if core_mean_intensity > config.grid_gap_max_core_mean_intensity:
        return False
    if dark_fraction < config.grid_gap_min_dark_pixel_fraction:
        return False
    if outer_dark_fraction < config.grid_gap_min_outer_dark_pixel_fraction:
        return False

    component_metrics = _dark_component_shape_metrics(
        image=image,
        x=x,
        y=y,
        radius=radius,
        config=config,
    )
    if component_metrics is None:
        return False

    component_aspect_ratio, component_area_ratio = component_metrics
    return (
        component_aspect_ratio <= config.grid_gap_max_component_aspect_ratio
        and component_area_ratio >= config.grid_gap_min_component_area_ratio
    )


def _refine_circle_boundaries(
    image: np.ndarray,
    circles: list[tuple[int, int, int]],
    config: HoughCircleConfig,
) -> list[tuple[int, int, int]]:
    """Expand Hough circles to the visible solder pad boundary."""
    return [
        (
            x,
            y,
            _estimate_pad_radius(image=image, x=x, y=y, radius=radius, config=config),
        )
        for x, y, radius in circles
    ]


def _estimate_pad_radius(
    image: np.ndarray,
    x: int,
    y: int,
    radius: int,
    config: HoughCircleConfig,
) -> int:
    """Estimate the outer pad radius from radial dark-to-bright transitions."""
    height, width = image.shape[:2]
    max_possible_radius = min(x, y, width - x - 1, height - y - 1)
    growth_cap_radius = _max_refined_radius(radius, max_possible_radius, config)
    min_search_radius = max(3, int(radius * config.boundary_min_scale))
    max_search_radius = min(
        max_possible_radius,
        growth_cap_radius,
        max(radius + 2, int(radius * config.boundary_search_scale)),
    )
    if max_search_radius <= min_search_radius + (config.boundary_window_size * 2):
        return radius

    angles = np.linspace(0, 2 * np.pi, config.boundary_angle_count, endpoint=False)
    candidates: list[int] = []
    window = max(1, config.boundary_window_size)

    for angle in angles:
        cos_angle = float(np.cos(angle))
        sin_angle = float(np.sin(angle))
        profile = []
        for sample_radius in range(0, max_search_radius + window + 1):
            sample_x = int(round(x + (cos_angle * sample_radius)))
            sample_y = int(round(y + (sin_angle * sample_radius)))
            if sample_x < 0 or sample_y < 0 or sample_x >= width or sample_y >= height:
                break
            profile.append(int(image[sample_y, sample_x]))

        if len(profile) <= max_search_radius + window:
            continue

        profile_array = np.array(profile, dtype=np.float32)
        best_radius = radius
        best_gradient = 0.0
        for search_radius in range(min_search_radius, max_search_radius - window):
            inside_start = max(0, search_radius - window)
            inside = profile_array[inside_start:search_radius]
            outside = profile_array[search_radius : search_radius + window]
            if inside.size == 0 or outside.size == 0:
                continue

            inside_mean = float(np.mean(inside))
            outside_mean = float(np.mean(outside))
            gradient = outside_mean - inside_mean
            if gradient > best_gradient:
                best_gradient = gradient
                best_radius = search_radius

        if best_gradient >= config.boundary_min_gradient:
            candidates.append(best_radius)

    min_valid = max(
        8,
        int(config.boundary_angle_count * config.boundary_min_valid_fraction),
    )
    if len(candidates) < min_valid:
        return radius

    refined_radius = int(np.percentile(candidates, config.boundary_radius_percentile))
    return max(radius, min(refined_radius, growth_cap_radius, max_search_radius))


def _max_refined_radius(
    radius: int,
    max_possible_radius: int,
    config: HoughCircleConfig,
) -> int:
    """Return an adaptive upper bound for boundary refinement growth."""
    if radius <= config.boundary_small_radius_threshold:
        growth_scale = config.boundary_small_radius_growth_scale
    elif radius <= config.boundary_medium_radius_threshold:
        growth_scale = config.boundary_medium_radius_growth_scale
    else:
        growth_scale = config.boundary_large_radius_growth_scale

    capped_radius = max(radius + 2, int(round(radius * growth_scale)))
    return min(max_possible_radius, capped_radius)
