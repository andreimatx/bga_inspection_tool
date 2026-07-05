from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ball_detection import (
    HoughCircleConfig,
    crop_ball_roi,
    detect_solder_balls_with_diagnostics,
)
from src.image_loader import list_image_files, load_grayscale
from src.metrics import ComponentFilterConfig, analyze_void_components
from src.preprocessing import PreprocessingConfig, preprocess_image
from src.reporting import build_summary, save_reports
from src.visualization import (
    annotate_detected_balls,
    create_void_mask_preview,
    draw_bga_roi_debug,
    draw_pad_contour_overlay,
    save_image,
)
from src.void_segmentation import VoidSegmentationConfig, segment_voids


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Examples:
        python src/main.py
        python src/main.py --image Images_BGA/Raw/PCBA2/PCBA2_05.jpg
        python src/main.py --input-dir Images_BGA/Raw/PCBA2
    """
    parser = argparse.ArgumentParser(
        description="BGA solder void inspection from X-ray images.",
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
        default=PROJECT_ROOT / "Images_BGA" / "Results",
        help="Root folder where result images and reports will be saved.",
    )

    parser.add_argument(
        "--mask-library-root",
        type=Path,
        default=PROJECT_ROOT / "Images_BGA" / "Masks",
        help="Dedicated folder for per-ball void masks.",
    )

    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=15.0,
        help="Void ratio percent above which a ball is marked as REVIEW.",
    )

    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=25.0,
        help="Void ratio percent above which a ball is marked as FAIL.",
    )

    parser.add_argument(
        "--largest-void-fail-threshold",
        type=float,
        default=12.0,
        help="Largest single void ratio above which a ball is marked as FAIL.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve absolute and project-relative paths."""
    if path.is_absolute():
        return path

    project_relative = PROJECT_ROOT / path

    if project_relative.exists():
        return project_relative

    return path.resolve()


def ask_input_path() -> tuple[str, Path]:
    """Ask for an image or folder path when no CLI input is provided."""
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


def classify_ball_quality(
    void_ratio_percent: float,
    largest_void_ratio_percent: float,
    warning_threshold: float,
    fail_threshold: float,
    largest_void_fail_threshold: float,
) -> str:
    """Classify one solder ball based on total and largest void ratio."""
    if void_ratio_percent >= fail_threshold:
        return "FAIL"

    if largest_void_ratio_percent >= largest_void_fail_threshold:
        return "FAIL"

    if void_ratio_percent >= warning_threshold:
        return "REVIEW"

    return "PASS"


def classify_image_quality(rows: list[dict[str, object]]) -> str:
    """Classify the whole inspected image/BGA."""
    qualities = [str(row["quality"]) for row in rows]

    if "FAIL" in qualities:
        return "FAIL"

    if "REVIEW" in qualities:
        return "REVIEW"

    return "PASS"


def create_output_folders(
    output_root: Path,
    mask_library_root: Path,
    image_stem: str,
) -> dict[str, Path]:
    """Create all output folders."""
    folders = {
        "processed": output_root / "Processed",
        "roi": output_root / "ROI" / image_stem,
        "masks": output_root / "Masks" / image_stem,
        "reports": output_root / "Reports",
        "mask_library": mask_library_root / image_stem,
    }

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return folders


def print_bga_roi_info(diagnostics) -> None:
    """Print detected BGA ROI information."""
    if diagnostics.bga_roi is None:
        print("[INFO] BGA ROI: not available")
        return

    roi = diagnostics.bga_roi
    print(
        "[INFO] BGA ROI: "
        f"x_min={roi.x_min}, y_min={roi.y_min}, "
        f"x_max={roi.x_max}, y_max={roi.y_max}",
    )


def inspect_image(
    image_path: Path,
    output_root: Path,
    mask_library_root: Path,
    warning_threshold: float,
    fail_threshold: float,
    largest_void_fail_threshold: float,
) -> None:
    """Run complete BGA void inspection for one image."""
    image_path = resolve_path(image_path)
    output_root = resolve_path(output_root)
    mask_library_root = resolve_path(mask_library_root)

    image_name = image_path.name
    image_stem = image_path.stem

    print("\n" + "=" * 80)
    print(f"[INFO] Processing image: {image_path}")
    print("=" * 80)

    folders = create_output_folders(
        output_root=output_root,
        mask_library_root=mask_library_root,
        image_stem=image_stem,
    )

    raw = load_grayscale(image_path)
    print(f"[INFO] Image loaded: {raw.shape[1]} x {raw.shape[0]} px")

    preprocessing_config = PreprocessingConfig()
    clahe_image, denoised = preprocess_image(raw, preprocessing_config)

    save_image(folders["processed"] / f"{image_stem}_01_clahe.jpg", clahe_image)
    save_image(folders["processed"] / f"{image_stem}_02_denoised.jpg", denoised)

    print("[INFO] Preprocessing completed: CLAHE + Median Filter")

    ball_config = HoughCircleConfig()
    detection_result = detect_solder_balls_with_diagnostics(denoised, ball_config)

    balls = detection_result.balls
    diagnostics = detection_result.diagnostics

    print_bga_roi_info(diagnostics)
    print(f"[INFO] Detected BGA solder balls / pads: {len(balls)}")
    print(
        "[INFO] Grid slots: "
        f"{diagnostics.occupied_grid_slots}/{diagnostics.expected_grid_slots}",
    )

    if diagnostics.missing_grid_positions:
        print(
            "[WARNING] Missing / weak grid positions: "
            f"{len(diagnostics.missing_grid_positions)}",
        )

    roi_debug = draw_bga_roi_debug(raw, diagnostics)
    save_image(
        folders["processed"] / f"{image_stem}_03_bga_roi_debug.jpg",
        roi_debug,
    )

    contour_overlay = draw_pad_contour_overlay(
        grayscale=raw,
        balls=balls,
        diagnostics=diagnostics,
        detection_image=denoised,
        include_roi=True,
    )
    save_image(
        folders["processed"] / f"{image_stem}_04_pad_contours.jpg",
        contour_overlay,
    )

    void_config = VoidSegmentationConfig()
    component_config = ComponentFilterConfig()

    rows: list[dict[str, object]] = []
    mask_entries = []

    print("[INFO] Creating per-ball circular void masks...")

    for ball in balls:
        roi_raw, bounds, local_center = crop_ball_roi(raw, ball)

        void_mask = segment_voids(
            roi=roi_raw,
            local_center=local_center,
            radius=ball.radius,
            config=void_config,
        )

        metrics, clean_mask = analyze_void_components(
            mask=void_mask,
            ball_area=ball.area,
            config=component_config,
            roi=roi_raw,
        )

        void_ratio_percent = (
            metrics.total_void_area / ball.area * 100.0
            if ball.area > 0
            else 0.0
        )

        largest_void_ratio_percent = (
            metrics.largest_void_area / ball.area * 100.0
            if ball.area > 0
            else 0.0
        )

        quality = classify_ball_quality(
            void_ratio_percent=void_ratio_percent,
            largest_void_ratio_percent=largest_void_ratio_percent,
            warning_threshold=warning_threshold,
            fail_threshold=fail_threshold,
            largest_void_fail_threshold=largest_void_fail_threshold,
        )

        rows.append(
            {
                "image_name": image_name,
                "ball_id": ball.ball_id,
                "center_x": ball.center_x,
                "center_y": ball.center_y,
                "diameter_px": ball.diameter,
                "ball_area_px": round(ball.area, 2),
                "void_count": metrics.void_count,
                "total_void_area_px": metrics.total_void_area,
                "largest_void_area_px": metrics.largest_void_area,
                "void_ratio_percent": round(void_ratio_percent, 4),
                "largest_void_ratio_percent": round(largest_void_ratio_percent, 4),
                "quality": quality,
                "is_estimated_pad": ball.is_estimated,
                "confidence": ball.confidence,
            },
        )

        mask_entries.append((bounds, clean_mask))

        roi_filename = f"{image_stem}_ball_{ball.ball_id:03d}_roi.jpg"
        mask_filename = f"{image_stem}_ball_{ball.ball_id:03d}_mask.jpg"

        save_image(folders["roi"] / roi_filename, roi_raw)
        save_image(folders["masks"] / mask_filename, clean_mask)

        # Dedicated mask library requested for visual validation.
        save_image(folders["mask_library"] / mask_filename, clean_mask)

    print("[INFO] Void mask generation completed")

    annotated = annotate_detected_balls(
        grayscale=raw,
        balls=balls,
        rows=rows,
        warning_threshold=warning_threshold,
    )

    save_image(
        folders["processed"] / f"{image_stem}_05_annotated_metrics.jpg",
        annotated,
    )

    void_preview = create_void_mask_preview(
        grayscale=raw,
        balls=balls,
        mask_entries=mask_entries,
    )

    save_image(
        folders["processed"] / f"{image_stem}_06_void_preview.jpg",
        void_preview,
    )

    summary = build_summary(
        image_name=image_name,
        rows=rows,
        warning_threshold=warning_threshold,
    )

    verdict = classify_image_quality(rows)

    summary["qualitative_verdict"] = verdict
    summary["fail_threshold_percent"] = fail_threshold
    summary["largest_void_fail_threshold_percent"] = largest_void_fail_threshold
    summary["balls_pass"] = sum(row["quality"] == "PASS" for row in rows)
    summary["balls_review"] = sum(row["quality"] == "REVIEW" for row in rows)
    summary["balls_fail"] = sum(row["quality"] == "FAIL" for row in rows)

    report_paths = save_reports(
        rows=rows,
        summary=summary,
        output_dir=folders["reports"],
        image_stem=image_stem,
    )

    print("\n[RESULT] Inspection completed")
    print(f"[RESULT] Image: {image_name}")
    print(f"[RESULT] Balls detected: {len(balls)}")
    print(f"[RESULT] Average void ratio: {summary['average_void_ratio_percent']}%")
    print(f"[RESULT] Maximum void ratio: {summary['maximum_void_ratio_percent']}%")
    print(f"[RESULT] PASS balls: {summary['balls_pass']}")
    print(f"[RESULT] REVIEW balls: {summary['balls_review']}")
    print(f"[RESULT] FAIL balls: {summary['balls_fail']}")
    print(f"[RESULT] Qualitative verdict: {verdict}")

    print("\n[OUTPUT] Processed images:")
    print(f"         {folders['processed']}")
    print("[OUTPUT] ROI crops:")
    print(f"         {folders['roi']}")
    print("[OUTPUT] Result masks:")
    print(f"         {folders['masks']}")
    print("[OUTPUT] Dedicated mask library:")
    print(f"         {folders['mask_library']}")
    print("[OUTPUT] Reports:")
    print(f"         {folders['reports']}")
    print(f"[OUTPUT] Excel report: {report_paths['excel']}")


def inspect_folder(
    input_dir: Path,
    output_root: Path,
    mask_library_root: Path,
    warning_threshold: float,
    fail_threshold: float,
    largest_void_fail_threshold: float,
) -> None:
    """Run inspection for all supported images in a folder."""
    input_dir = resolve_path(input_dir)
    image_paths = list_image_files(input_dir)

    if not image_paths:
        raise RuntimeError(f"No supported images found in: {input_dir}")

    print(f"\n[INFO] Found {len(image_paths)} image(s) in: {input_dir}")

    for image_path in image_paths:
        inspect_image(
            image_path=image_path,
            output_root=output_root,
            mask_library_root=mask_library_root,
            warning_threshold=warning_threshold,
            fail_threshold=fail_threshold,
            largest_void_fail_threshold=largest_void_fail_threshold,
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
        inspect_image(
            image_path=image_path,
            output_root=args.output_root,
            mask_library_root=args.mask_library_root,
            warning_threshold=args.warning_threshold,
            fail_threshold=args.fail_threshold,
            largest_void_fail_threshold=args.largest_void_fail_threshold,
        )
        return

    if input_dir is not None:
        inspect_folder(
            input_dir=input_dir,
            output_root=args.output_root,
            mask_library_root=args.mask_library_root,
            warning_threshold=args.warning_threshold,
            fail_threshold=args.fail_threshold,
            largest_void_fail_threshold=args.largest_void_fail_threshold,
        )
        return

    raise RuntimeError("No image or input directory was selected.")


if __name__ == "__main__":
    main()