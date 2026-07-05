from __future__ import annotations

import argparse
import shutil
import sys
import traceback
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


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------
def _supports_unicode(stream) -> bool:
    """Return True if the stream can encode the fancy bar characters."""
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        "█░•".encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


class ProgressBar:
    """Minimal dependency-free progress bar for the inspection pipeline.

    It animates a single, in-place line on real terminals and degrades
    gracefully (plain log lines, no animation) when output is redirected to a
    file or runs in an environment that does not support carriage-return
    animation. It also falls back to ASCII characters when the console encoding
    cannot display the Unicode block characters (common on some Windows
    consoles).
    """

    def __init__(
        self,
        label: str = "Progress",
        total: float = 100.0,
        width: int = 28,
        mode: str = "auto",
        stream=None,
    ) -> None:
        self.label = label
        self.total = float(total) if total > 0 else 1.0
        self.width = max(int(width), 8)
        self.stream = stream if stream is not None else sys.stdout

        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        if mode == "on":
            self.enabled = True
        elif mode == "off":
            self.enabled = False
        else:  # "auto"
            self.enabled = is_tty

        if _supports_unicode(self.stream):
            self._fill, self._empty, self._sep = "█", "░", "•"
        else:
            self._fill, self._empty, self._sep = "#", "-", "-"

        self.current = 0.0
        self.status = ""
        self._last_len = 0
        self._finished = False

    def _render(self) -> None:
        if not self.enabled:
            return

        columns = shutil.get_terminal_size(fallback=(80, 24)).columns
        fraction = min(max(self.current / self.total, 0.0), 1.0)
        filled = int(round(fraction * self.width))
        bar = self._fill * filled + self._empty * (self.width - filled)
        percent = int(round(fraction * 100))

        prefix = f"{self.label} |"
        suffix = f"| {percent:3d}%"
        status = f" {self._sep} {self.status}" if self.status else ""

        line = f"{prefix}{bar}{suffix}{status}"
        # Keep everything on a single line so the carriage-return animation
        # does not wrap and spawn extra lines on narrow terminals.
        if len(line) > columns - 1:
            line = line[: max(columns - 1, 1)]

        pad = max(self._last_len - len(line), 0)
        self.stream.write("\r" + line + " " * pad)
        self.stream.flush()
        self._last_len = len(line)

    def set_progress(self, value: float, status: str | None = None) -> None:
        """Set the absolute progress value (clamped to the configured total)."""
        if status is not None:
            self.status = status
        self.current = min(max(float(value), 0.0), self.total)
        self._render()

    def update(self, amount: float = 0.0, status: str | None = None) -> None:
        """Advance the bar by a relative amount."""
        if status is not None:
            self.status = status
        self.current = min(self.current + float(amount), self.total)
        self._render()

    def log(self, message: str) -> None:
        """Print a normal log line without destroying the progress bar."""
        if self.enabled and not self._finished:
            # Erase the bar, print the message above it, then redraw the bar.
            self.stream.write("\r" + " " * self._last_len + "\r")
            self.stream.flush()
            print(message)
            self._last_len = 0
            self._render()
        else:
            print(message)

    def clear(self) -> None:
        """Erase the current bar line (used before printing an error)."""
        if self.enabled and not self._finished:
            self.stream.write("\r" + " " * self._last_len + "\r")
            self.stream.flush()
            self._last_len = 0

    def finish(self, status: str | None = None) -> None:
        """Complete the bar at 100% and move the cursor to a new line."""
        if self._finished:
            return
        if status is not None:
            self.status = status
        self.current = self.total
        self._render()
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()
        self._finished = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
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

    parser.add_argument(
        "--progress",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Show the processing progress bar: 'auto' (only in an interactive "
            "terminal), 'on' (force), or 'off' (disable)."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full Python tracebacks when an error occurs.",
    )

    return parser.parse_args()


def validate_thresholds(args: argparse.Namespace) -> None:
    """Sanity-check user-provided thresholds before doing any heavy work."""
    named_values = (
        ("--warning-threshold", args.warning_threshold),
        ("--fail-threshold", args.fail_threshold),
        ("--largest-void-fail-threshold", args.largest_void_fail_threshold),
    )

    for name, value in named_values:
        if value < 0:
            raise ValueError(f"{name} must be >= 0 (got {value}).")

    if args.warning_threshold > args.fail_threshold:
        print(
            "[WARNING] warning-threshold is greater than fail-threshold; "
            "balls will jump straight to FAIL without ever being REVIEW.",
        )


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
        try:
            user_input = input("Input path: ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D / Ctrl+Z / Ctrl+C at the prompt: quit cleanly.
            print("\n[INFO] Cancelled by user.")
            raise SystemExit(0)

        if user_input.lower() in {"q", "quit", "exit"}:
            print("[INFO] Cancelled by user.")
            raise SystemExit(0)

        if not user_input:
            print("[ERROR] Empty path. Please paste an image or folder path.")
            continue

        try:
            input_path = resolve_path(Path(user_input))
        except (OSError, ValueError) as exc:
            print(f"[ERROR] Invalid path '{user_input}': {exc}")
            continue

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


def format_bga_roi_info(diagnostics) -> str:
    """Return a printable line describing the detected BGA ROI."""
    if diagnostics.bga_roi is None:
        return "[INFO] BGA ROI: not available"

    roi = diagnostics.bga_roi
    return (
        "[INFO] BGA ROI: "
        f"x_min={roi.x_min}, y_min={roi.y_min}, "
        f"x_max={roi.x_max}, y_max={roi.y_max}"
    )


def inspect_image(
    image_path: Path,
    output_root: Path,
    mask_library_root: Path,
    warning_threshold: float,
    fail_threshold: float,
    largest_void_fail_threshold: float,
    progress_mode: str = "auto",
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

    progress = ProgressBar(label=image_stem, total=100.0, mode=progress_mode)
    progress.set_progress(0, "Starting")

    try:
        # --- Output folders -------------------------------------------------
        try:
            folders = create_output_folders(
                output_root=output_root,
                mask_library_root=mask_library_root,
                image_stem=image_stem,
            )
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot create output folders under '{output_root}'. Make "
                "sure no result folder is open in another program and that you "
                f"have write permission. ({exc})"
            ) from exc
        except OSError as exc:
            raise OSError(
                f"Could not create output folders under '{output_root}': {exc}"
            ) from exc

        progress.set_progress(3, "Output folders ready")

        # --- Load image -----------------------------------------------------
        try:
            raw = load_grayscale(image_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Image file not found: {image_path}") from exc
        except Exception as exc:  # noqa: BLE001 - normalize loader errors
            raise RuntimeError(
                f"Could not read image '{image_name}'. It may be corrupt, "
                f"empty, or in an unsupported format. ({exc})"
            ) from exc

        progress.set_progress(8, "Image loaded")
        progress.log(f"[INFO] Image loaded: {raw.shape[1]} x {raw.shape[0]} px")

        # --- Preprocessing --------------------------------------------------
        preprocessing_config = PreprocessingConfig()
        clahe_image, denoised = preprocess_image(raw, preprocessing_config)

        save_image(folders["processed"] / f"{image_stem}_01_clahe.jpg", clahe_image)
        save_image(folders["processed"] / f"{image_stem}_02_denoised.jpg", denoised)

        progress.set_progress(18, "Preprocessing done")
        progress.log("[INFO] Preprocessing completed: CLAHE + Median Filter")

        # --- Ball detection -------------------------------------------------
        ball_config = HoughCircleConfig()
        detection_result = detect_solder_balls_with_diagnostics(denoised, ball_config)

        balls = detection_result.balls
        diagnostics = detection_result.diagnostics

        progress.set_progress(30, "Balls detected")
        progress.log(format_bga_roi_info(diagnostics))
        progress.log(f"[INFO] Detected BGA solder balls / pads: {len(balls)}")
        progress.log(
            "[INFO] Grid slots: "
            f"{diagnostics.occupied_grid_slots}/{diagnostics.expected_grid_slots}"
        )

        if diagnostics.missing_grid_positions:
            progress.log(
                "[WARNING] Missing / weak grid positions: "
                f"{len(diagnostics.missing_grid_positions)}"
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

        progress.set_progress(38, "Debug overlays saved")

        # --- Per-ball void segmentation (heaviest stage) --------------------
        void_config = VoidSegmentationConfig()
        component_config = ComponentFilterConfig()

        rows: list[dict[str, object]] = []
        mask_entries = []

        n_balls = len(balls)
        progress.log("[INFO] Creating per-ball circular void masks...")
        progress.set_progress(38, "Segmenting voids")

        # The void loop spans the 38% -> 85% portion of the bar.
        per_ball = (85.0 - 38.0) / n_balls if n_balls > 0 else 0.0

        for index, ball in enumerate(balls, start=1):
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

            progress.update(
                per_ball,
                f"Segmenting voids (ball {index}/{n_balls})",
            )

        progress.set_progress(85, "Void masks complete")
        progress.log("[INFO] Void mask generation completed")

        # --- Annotated outputs ----------------------------------------------
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

        progress.set_progress(93, "Annotations saved")

        # --- Summary + reports ----------------------------------------------
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

        progress.set_progress(100, "Inspection complete")
        progress.finish("Inspection complete")

        # --- Final report (printed after the bar has finished) --------------
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
    finally:
        # If we exited via an exception, finish() never ran: erase the partial
        # bar so the error message starts on a clean line.
        progress.clear()


def inspect_folder(
    input_dir: Path,
    output_root: Path,
    mask_library_root: Path,
    warning_threshold: float,
    fail_threshold: float,
    largest_void_fail_threshold: float,
    progress_mode: str = "auto",
    debug: bool = False,
) -> None:
    """Run inspection for all supported images in a folder.

    A failure on one image is logged and the batch continues, so a single bad
    or locked file never aborts the whole run.
    """
    input_dir = resolve_path(input_dir)

    try:
        image_paths = list_image_files(input_dir)
    except Exception as exc:  # noqa: BLE001 - normalize listing errors
        raise RuntimeError(f"Could not list images in '{input_dir}': {exc}") from exc

    if not image_paths:
        raise RuntimeError(f"No supported images found in: {input_dir}")

    total = len(image_paths)
    print(f"\n[INFO] Found {total} image(s) in: {input_dir}")

    succeeded = 0
    failures: list[tuple[Path, Exception]] = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"\n[INFO] ----- Image {index}/{total} -----")
        try:
            inspect_image(
                image_path=image_path,
                output_root=output_root,
                mask_library_root=mask_library_root,
                warning_threshold=warning_threshold,
                fail_threshold=fail_threshold,
                largest_void_fail_threshold=largest_void_fail_threshold,
                progress_mode=progress_mode,
            )
            succeeded += 1
        except KeyboardInterrupt:
            # Let the user abort the entire batch with Ctrl+C.
            print("\n[ABORTED] Batch interrupted by user (Ctrl+C).")
            raise
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            failures.append((image_path, exc))
            print(f"[ERROR] Skipped '{image_path.name}': {exc}")
            if debug:
                traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"[BATCH] Completed: {succeeded}/{total} image(s) processed successfully.")
    if failures:
        print(f"[BATCH] Failed: {len(failures)} image(s):")
        for path, exc in failures:
            print(f"        - {path.name}: {exc}")
    print("=" * 80)


def _print_error(title: str, exc: BaseException, debug: bool) -> None:
    """Print a friendly error message, with an optional full traceback."""
    print(f"\n[ERROR] {title}: {exc}")
    if debug:
        traceback.print_exc()
    else:
        print("[HINT] Run again with --debug to see the full traceback.")


def _run(args: argparse.Namespace) -> None:
    """Resolve the input and dispatch to single-image or folder inspection."""
    validate_thresholds(args)

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
            progress_mode=args.progress,
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
            progress_mode=args.progress,
            debug=args.debug,
        )
        return

    raise RuntimeError("No image or input directory was selected.")


def main() -> None:
    """Application entry point with top-level error handling."""
    # Make printing resilient to console encodings that cannot render some
    # characters (e.g. diacritics in file paths on Windows), instead of
    # crashing with a UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    args = parse_args()
    debug = args.debug

    try:
        _run(args)
    except KeyboardInterrupt:
        print("\n[ABORTED] Interrupted by user (Ctrl+C).")
        raise SystemExit(130)
    except SystemExit:
        # Deliberate exits (e.g. user typed 'q') propagate unchanged.
        raise
    except ValueError as exc:
        _print_error("Invalid configuration", exc, debug)
        raise SystemExit(2)
    except FileNotFoundError as exc:
        _print_error("File or folder not found", exc, debug)
        raise SystemExit(1)
    except PermissionError as exc:
        _print_error(
            "Permission denied — a file or folder may be open in another "
            "program or read-only",
            exc,
            debug,
        )
        raise SystemExit(1)
    except RuntimeError as exc:
        _print_error("Processing error", exc, debug)
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        _print_error("Unexpected error", exc, debug)
        raise SystemExit(1)


if __name__ == "__main__":
    main()