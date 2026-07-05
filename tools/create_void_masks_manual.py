from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ball_detection import (
    HoughCircleConfig,
    crop_ball_roi,
    detect_solder_balls_with_diagnostics,
)
from src.image_loader import list_image_files, load_grayscale
from src.preprocessing import PreprocessingConfig, preprocess_image
from src.visualization import (
    create_void_mask_preview,
    draw_pad_contour_overlay,
)


@dataclass
class CircleAnnotation:
    """One manually annotated void circle."""

    x: int
    y: int
    radius: int


@dataclass
class AnnotationState:
    """Interactive annotation state for one solder ball ROI."""

    image_stem: str
    ball_id: int
    roi: np.ndarray
    local_center: tuple[int, int]
    pad_radius: int
    output_dir: Path
    circles: list[CircleAnnotation] = field(default_factory=list)
    brush_radius: int = 7
    zoom: int = 5
    show_overlay: bool = True
    saved: bool = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Manual/semi-manual void mask creator for BGA solder balls.",
    )

    input_group = parser.add_mutually_exclusive_group(required=False)

    input_group.add_argument(
        "--image",
        type=Path,
        help="Path to one X-ray image.",
    )

    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Path to a folder containing X-ray images.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "Images_BGA" / "Masks_Manual",
        help="Folder where manual masks will be saved.",
    )

    parser.add_argument(
        "--start-ball",
        type=int,
        default=1,
        help="First ball id to annotate.",
    )

    parser.add_argument(
        "--zoom",
        type=int,
        default=5,
        help="Display zoom factor.",
    )

    parser.add_argument(
        "--default-radius",
        type=int,
        default=7,
        help="Default void brush radius in pixels.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve absolute or project-relative path."""
    if path.is_absolute():
        return path

    project_relative = PROJECT_ROOT / path
    if project_relative.exists():
        return project_relative

    return path.resolve()


def ask_input_path() -> tuple[str, Path]:
    """Ask the user for an image or folder path."""
    print("\nNo input image/folder was provided.")
    print("Paste the path to an X-ray image or to a folder with images.")
    print("\nExample image:")
    print(r"D:\BGA_Inspection_Claude\Images_BGA\Raw\PCBA2\PCBA2_05.jpg")
    print("\nExample folder:")
    print(r"D:\BGA_Inspection_Claude\Images_BGA\Raw\PCBA2")
    print("\nType 'q' to quit.\n")

    while True:
        user_input = input("Input path: ").strip().strip('"').strip("'")

        if user_input.lower() in {"q", "quit", "exit"}:
            raise SystemExit("User cancelled.")

        if not user_input:
            print("[ERROR] Empty path. Please paste an image or folder path.")
            continue

        input_path = resolve_path(Path(user_input))

        if not input_path.exists():
            print(f"[ERROR] Path does not exist: {input_path}")
            continue

        if input_path.is_file():
            return "image", input_path

        if input_path.is_dir():
            return "input_dir", input_path

        print(f"[ERROR] Invalid path: {input_path}")


def print_controls() -> None:
    """Print interactive controls."""
    print("\n" + "=" * 80)
    print("MANUAL VOID MASK CREATOR - controls")
    print("=" * 80)
    print("Left click       : add void circle")
    print("Right click      : remove nearest circle")
    print("1..9             : set radius = key * 2 px")
    print("[ / ]            : decrease / increase radius")
    print("u                : undo last circle")
    print("c                : clear current ball")
    print("v                : toggle red overlay")
    print("s                : save current ball and go next")
    print("e                : save empty mask and go next")
    print("n                : skip current ball without saving")
    print("q / ESC          : quit")
    print("=" * 80 + "\n")


def load_annotations(annotation_path: Path) -> dict[str, list[dict[str, int]]]:
    """Load previous annotations if they exist."""
    if not annotation_path.exists():
        return {}

    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        return {}

    return data


def save_annotations(
    annotation_path: Path,
    annotations: dict[str, list[dict[str, int]]],
) -> None:
    """Save annotations to JSON."""
    annotation_path.parent.mkdir(parents=True, exist_ok=True)

    with annotation_path.open("w", encoding="utf-8") as file:
        json.dump(annotations, file, indent=2)


def circles_from_json(items: list[dict[str, int]]) -> list[CircleAnnotation]:
    """Convert JSON circles to dataclass circles."""
    circles: list[CircleAnnotation] = []

    for item in items:
        try:
            circles.append(
                CircleAnnotation(
                    x=int(item["x"]),
                    y=int(item["y"]),
                    radius=int(item["radius"]),
                ),
            )
        except (KeyError, TypeError, ValueError):
            continue

    return circles


def circles_to_json(circles: list[CircleAnnotation]) -> list[dict[str, int]]:
    """Convert dataclass circles to JSON serializable dictionaries."""
    return [
        {
            "x": int(circle.x),
            "y": int(circle.y),
            "radius": int(circle.radius),
        }
        for circle in circles
    ]


def build_mask_from_circles(
    roi_shape: tuple[int, int],
    circles: list[CircleAnnotation],
    local_center: tuple[int, int],
    pad_radius: int,
) -> np.ndarray:
    """Create a binary mask from manually annotated circles."""
    mask = np.zeros(roi_shape, dtype=np.uint8)

    pad_mask = np.zeros(roi_shape, dtype=np.uint8)
    inspected_radius = max(1, int(round(pad_radius * 0.90)))
    cv2.circle(pad_mask, local_center, inspected_radius, 255, thickness=-1)

    for circle in circles:
        cv2.circle(
            mask,
            (int(circle.x), int(circle.y)),
            int(circle.radius),
            255,
            thickness=-1,
        )

    return cv2.bitwise_and(mask, pad_mask)


def render_annotation_view(state: AnnotationState) -> np.ndarray:
    """Render current ROI with green pad contour and red void annotations."""
    if state.roi.ndim == 2:
        view = cv2.cvtColor(state.roi, cv2.COLOR_GRAY2BGR)
    else:
        view = state.roi.copy()

    pad_radius = max(1, int(round(state.pad_radius * 0.90)))
    cv2.circle(
        view,
        state.local_center,
        pad_radius,
        (0, 180, 0),
        thickness=1,
    )

    mask = build_mask_from_circles(
        roi_shape=state.roi.shape[:2],
        circles=state.circles,
        local_center=state.local_center,
        pad_radius=state.pad_radius,
    )

    if state.show_overlay:
        red_layer = view.copy()
        red_layer[mask > 0] = (0, 0, 255)
        view = cv2.addWeighted(view, 0.72, red_layer, 0.28, 0)

    for circle in state.circles:
        cv2.circle(
            view,
            (circle.x, circle.y),
            circle.radius,
            (0, 0, 255),
            thickness=1,
        )

    cv2.putText(
        view,
        f"Ball {state.ball_id:03d} | radius={state.brush_radius}px | "
        f"voids={len(state.circles)}",
        (5, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    zoom = max(1, int(state.zoom))
    return cv2.resize(
        view,
        None,
        fx=zoom,
        fy=zoom,
        interpolation=cv2.INTER_NEAREST,
    )


def point_inside_pad(
    x: int,
    y: int,
    local_center: tuple[int, int],
    pad_radius: int,
) -> bool:
    """Check whether a point is inside the inspected pad region."""
    distance = float(np.hypot(x - local_center[0], y - local_center[1]))
    return distance <= float(pad_radius) * 0.90


def remove_nearest_circle(
    circles: list[CircleAnnotation],
    x: int,
    y: int,
) -> list[CircleAnnotation]:
    """Remove the nearest circle to a clicked point."""
    if not circles:
        return circles

    distances = [
        float(np.hypot(circle.x - x, circle.y - y))
        for circle in circles
    ]

    nearest_index = int(np.argmin(distances))
    nearest_circle = circles[nearest_index]

    if distances[nearest_index] <= max(8.0, nearest_circle.radius * 1.6):
        circles.pop(nearest_index)

    return circles


def mouse_callback(
    event: int,
    x: int,
    y: int,
    flags: int,
    state: AnnotationState,
) -> None:
    """Handle mouse events in the annotation window."""
    del flags

    roi_x = int(round(x / state.zoom))
    roi_y = int(round(y / state.zoom))

    if event == cv2.EVENT_LBUTTONDOWN:
        if point_inside_pad(roi_x, roi_y, state.local_center, state.pad_radius):
            state.circles.append(
                CircleAnnotation(
                    x=roi_x,
                    y=roi_y,
                    radius=max(1, int(state.brush_radius)),
                ),
            )

    elif event == cv2.EVENT_RBUTTONDOWN:
        state.circles = remove_nearest_circle(state.circles, roi_x, roi_y)


def save_current_ball(
    state: AnnotationState,
    annotations: dict[str, list[dict[str, int]]],
) -> tuple[Path, Path, Path]:
    """Save current ROI, mask and overlay."""
    ball_key = f"{state.ball_id:03d}"

    mask = build_mask_from_circles(
        roi_shape=state.roi.shape[:2],
        circles=state.circles,
        local_center=state.local_center,
        pad_radius=state.pad_radius,
    )

    overlay = render_annotation_view(state)
    overlay_original_size = cv2.resize(
        overlay,
        (state.roi.shape[1], state.roi.shape[0]),
        interpolation=cv2.INTER_AREA,
    )

    roi_path = state.output_dir / f"{state.image_stem}_ball_{ball_key}_roi.png"
    mask_path = state.output_dir / f"{state.image_stem}_ball_{ball_key}_mask.png"
    overlay_path = state.output_dir / f"{state.image_stem}_ball_{ball_key}_overlay.png"

    cv2.imwrite(str(roi_path), state.roi)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(overlay_path), overlay_original_size)

    annotations[ball_key] = circles_to_json(state.circles)

    state.saved = True

    return roi_path, mask_path, overlay_path


def annotate_one_ball(
    state: AnnotationState,
    annotations: dict[str, list[dict[str, int]]],
) -> str:
    """Interactive annotation loop for one solder ball."""
    window_name = f"{state.image_stem} - ball {state.ball_id:03d}"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback, state)

    while True:
        view = render_annotation_view(state)
        cv2.imshow(window_name, view)

        key = cv2.waitKey(30) & 0xFF

        if key == 255:
            continue

        if key in {27, ord("q")}:
            cv2.destroyWindow(window_name)
            return "quit"

        if ord("1") <= key <= ord("9"):
            state.brush_radius = (key - ord("0")) * 2

        elif key == ord("["):
            state.brush_radius = max(1, state.brush_radius - 1)

        elif key == ord("]"):
            state.brush_radius = state.brush_radius + 1

        elif key == ord("u"):
            if state.circles:
                state.circles.pop()

        elif key == ord("c"):
            state.circles.clear()

        elif key == ord("v"):
            state.show_overlay = not state.show_overlay

        elif key == ord("e"):
            state.circles.clear()
            save_current_ball(state, annotations)
            cv2.destroyWindow(window_name)
            return "next"

        elif key == ord("s"):
            save_current_ball(state, annotations)
            cv2.destroyWindow(window_name)
            return "next"

        elif key == ord("n"):
            cv2.destroyWindow(window_name)
            return "next"


def save_full_manual_preview(
    raw: np.ndarray,
    image_stem: str,
    balls: list,
    ball_data: list[tuple[object, tuple, np.ndarray]],
    output_dir: Path,
) -> Path:
    """Save a full-board preview using the manually created masks."""
    mask_entries = [(bounds, mask) for _, bounds, mask in ball_data]

    preview = create_void_mask_preview(
        grayscale=raw,
        balls=balls,
        mask_entries=mask_entries,
    )

    preview_path = output_dir / f"{image_stem}_manual_void_preview.png"
    cv2.imwrite(str(preview_path), preview)

    return preview_path


def process_image(
    image_path: Path,
    output_root: Path,
    start_ball: int,
    zoom: int,
    default_radius: int,
) -> None:
    """Detect pads and manually annotate void masks pad by pad."""
    image_path = resolve_path(image_path)
    output_root = resolve_path(output_root)

    image_stem = image_path.stem
    output_dir = output_root / image_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = output_dir / f"{image_stem}_annotations.json"
    annotations = load_annotations(annotation_path)

    print("\n" + "=" * 80)
    print(f"[INFO] Processing image: {image_path}")
    print(f"[INFO] Output folder: {output_dir}")
    print("=" * 80)

    raw = load_grayscale(image_path)

    preprocessing_config = PreprocessingConfig()
    _, denoised = preprocess_image(raw, preprocessing_config)

    ball_config = HoughCircleConfig()
    detection_result = detect_solder_balls_with_diagnostics(denoised, ball_config)

    balls = detection_result.balls
    diagnostics = detection_result.diagnostics

    print(f"[INFO] Detected pads: {len(balls)}")
    print(
        "[INFO] Grid slots: "
        f"{diagnostics.occupied_grid_slots}/{diagnostics.expected_grid_slots}",
    )

    contour_overlay = draw_pad_contour_overlay(
        grayscale=raw,
        balls=balls,
        diagnostics=diagnostics,
        detection_image=denoised,
        include_roi=True,
    )

    contour_path = output_dir / f"{image_stem}_pad_contours.png"
    cv2.imwrite(str(contour_path), contour_overlay)
    print(f"[INFO] Pad contour preview saved: {contour_path}")

    print_controls()

    ball_data: list[tuple[object, tuple, np.ndarray]] = []

    for ball in balls:
        ball_id = int(ball.ball_id)

        roi_raw, bounds, local_center = crop_ball_roi(raw, ball)

        ball_key = f"{ball_id:03d}"
        existing_circles = circles_from_json(annotations.get(ball_key, []))

        mask_path = output_dir / f"{image_stem}_ball_{ball_key}_mask.png"
        if mask_path.exists() and existing_circles:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros_like(roi_raw, dtype=np.uint8)
        else:
            mask = np.zeros_like(roi_raw, dtype=np.uint8)

        ball_data.append((ball, bounds, mask))

    for index, ball in enumerate(balls):
        ball_id = int(ball.ball_id)

        if ball_id < start_ball:
            continue

        roi_raw, bounds, local_center = crop_ball_roi(raw, ball)

        ball_key = f"{ball_id:03d}"
        existing_circles = circles_from_json(annotations.get(ball_key, []))

        state = AnnotationState(
            image_stem=image_stem,
            ball_id=ball_id,
            roi=roi_raw,
            local_center=local_center,
            pad_radius=int(ball.radius),
            output_dir=output_dir,
            circles=existing_circles,
            brush_radius=default_radius,
            zoom=zoom,
        )

        print(
            f"[INFO] Ball {ball_id:03d}/{len(balls):03d} - "
            f"existing voids: {len(existing_circles)}",
        )

        action = annotate_one_ball(state, annotations)

        save_annotations(annotation_path, annotations)

        saved_mask_path = output_dir / f"{image_stem}_ball_{ball_key}_mask.png"
        saved_mask = cv2.imread(str(saved_mask_path), cv2.IMREAD_GRAYSCALE)

        if saved_mask is None:
            saved_mask = np.zeros_like(roi_raw, dtype=np.uint8)

        ball_data[index] = (ball, bounds, saved_mask)

        if action == "quit":
            print("[INFO] Annotation stopped by user.")
            break

    save_annotations(annotation_path, annotations)

    preview_path = save_full_manual_preview(
        raw=raw,
        image_stem=image_stem,
        balls=balls,
        ball_data=ball_data,
        output_dir=output_dir,
    )

    print(f"[INFO] Manual annotations saved: {annotation_path}")
    print(f"[INFO] Full manual preview saved: {preview_path}")
    print("[INFO] Done.")


def process_folder(
    input_dir: Path,
    output_root: Path,
    start_ball: int,
    zoom: int,
    default_radius: int,
) -> None:
    """Process all images from a folder."""
    input_dir = resolve_path(input_dir)
    image_paths = list_image_files(input_dir)

    if not image_paths:
        raise RuntimeError(f"No supported images found in: {input_dir}")

    print(f"[INFO] Found {len(image_paths)} image(s) in: {input_dir}")

    for image_path in image_paths:
        process_image(
            image_path=image_path,
            output_root=output_root,
            start_ball=start_ball,
            zoom=zoom,
            default_radius=default_radius,
        )


def main() -> None:
    """Application entry point."""
    args = parse_args()

    image_path = args.image
    input_dir = args.input_dir

    if image_path is None and input_dir is None:
        input_type, selected_path = ask_input_path()

        if input_type == "image":
            image_path = selected_path
        else:
            input_dir = selected_path

    if image_path is not None:
        process_image(
            image_path=image_path,
            output_root=args.output_root,
            start_ball=args.start_ball,
            zoom=args.zoom,
            default_radius=args.default_radius,
        )
        return

    if input_dir is not None:
        process_folder(
            input_dir=input_dir,
            output_root=args.output_root,
            start_ball=args.start_ball,
            zoom=args.zoom,
            default_radius=args.default_radius,
        )
        return

    raise RuntimeError("No image or input directory was selected.")


if __name__ == "__main__":
    main()