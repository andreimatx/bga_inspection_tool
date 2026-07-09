"""Regression check for the BGA void inspection pipeline.

Runs the full analysis (detection + segmentation + metrics, no image saving)
on the reference X-ray images and compares the results with a saved baseline.
Use it after any parameter or algorithm change to see immediately whether
something got worse.

Usage:
    python tools/regression_check.py            # compare against baseline
    python tools/regression_check.py --update   # (re)write the baseline
    python tools/regression_check.py --images Images_BGA/Raw/PCBA1/PCBA1_07.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ball_detection import (  # noqa: E402
    HoughCircleConfig,
    crop_ball_roi,
    detect_solder_balls_with_diagnostics,
)
from src.image_loader import load_grayscale  # noqa: E402
from src.main import classify_ball_quality  # noqa: E402
from src.metrics import ComponentFilterConfig, analyze_void_components  # noqa: E402
from src.preprocessing import PreprocessingConfig, preprocess_image  # noqa: E402
from src.void_segmentation import VoidSegmentationConfig, segment_voids  # noqa: E402

BASELINE_PATH = PROJECT_ROOT / "tools" / "regression_baseline.json"

# Thresholds mirror the src/main.py CLI defaults (IPC-A-610 / IPC-7095).
WARNING_THRESHOLD = 10.0
FAIL_THRESHOLD = 25.0
LARGEST_VOID_FAIL_THRESHOLD = 12.25

# Comparison tolerances.
AVG_RATIO_TOLERANCE = 0.30       # percentage points on the image average
MAX_RATIO_TOLERANCE = 1.00       # percentage points on the image maximum
BALL_RATIO_TOLERANCE = 1.50      # percentage points per individual ball
MAX_DEVIATING_BALLS = 3          # how many balls may exceed the tolerance


def default_reference_images() -> list[Path]:
    """Return every raw reference image, excluding demo copies."""
    raw_root = PROJECT_ROOT / "Images_BGA" / "Raw"
    images = sorted(
        path
        for path in raw_root.rglob("*.jpg")
        if "DEMO" not in path.stem.upper()
    )
    return images


def analyze_image(image_path: Path) -> dict[str, object]:
    """Run the full analysis pipeline for one image and return its metrics."""
    raw = load_grayscale(image_path)
    _, denoised = preprocess_image(raw, PreprocessingConfig())

    detection = detect_solder_balls_with_diagnostics(denoised, HoughCircleConfig())
    balls = detection.balls

    void_config = VoidSegmentationConfig()
    component_config = ComponentFilterConfig()

    ball_ratios: dict[str, float] = {}
    qualities = {"PASS": 0, "REVIEW": 0, "FAIL": 0}

    for ball in balls:
        roi_raw, _, local_center = crop_ball_roi(raw, ball)
        void_mask = segment_voids(
            roi=roi_raw,
            local_center=local_center,
            radius=ball.radius,
            config=void_config,
        )
        metrics, _ = analyze_void_components(
            mask=void_mask,
            ball_area=ball.area,
            config=component_config,
            roi=roi_raw,
        )

        void_ratio = (
            metrics.total_void_area / ball.area * 100.0 if ball.area > 0 else 0.0
        )
        largest_ratio = (
            metrics.largest_void_area / ball.area * 100.0 if ball.area > 0 else 0.0
        )
        quality = classify_ball_quality(
            void_ratio_percent=void_ratio,
            largest_void_ratio_percent=largest_ratio,
            warning_threshold=WARNING_THRESHOLD,
            fail_threshold=FAIL_THRESHOLD,
            largest_void_fail_threshold=LARGEST_VOID_FAIL_THRESHOLD,
        )
        qualities[quality] += 1

        # Key by grid position, not ball_id: ids can shift when detection
        # order changes even though the physical pad is the same.
        key = f"{ball.center_x}x{ball.center_y}"
        ball_ratios[key] = round(void_ratio, 3)

    ratios = list(ball_ratios.values())
    return {
        "balls_detected": len(balls),
        "estimated_pads": sum(ball.is_estimated for ball in balls),
        "average_void_ratio": round(sum(ratios) / len(ratios), 3) if ratios else 0.0,
        "maximum_void_ratio": round(max(ratios), 3) if ratios else 0.0,
        "balls_pass": qualities["PASS"],
        "balls_review": qualities["REVIEW"],
        "balls_fail": qualities["FAIL"],
        "ball_ratios": ball_ratios,
    }


def compare_image_results(
    name: str,
    baseline: dict[str, object],
    current: dict[str, object],
) -> list[str]:
    """Compare one image's current metrics with the baseline.

    Returns a list of human-readable problems (empty list means OK).
    """
    problems: list[str] = []

    if current["balls_detected"] != baseline["balls_detected"]:
        problems.append(
            f"balls_detected: {baseline['balls_detected']} -> "
            f"{current['balls_detected']}"
        )

    for field, tolerance in (
        ("average_void_ratio", AVG_RATIO_TOLERANCE),
        ("maximum_void_ratio", MAX_RATIO_TOLERANCE),
    ):
        delta = abs(float(current[field]) - float(baseline[field]))
        if delta > tolerance:
            problems.append(
                f"{field}: {baseline[field]} -> {current[field]} "
                f"(delta {delta:.3f} > {tolerance})"
            )

    for field in ("balls_review", "balls_fail"):
        if current[field] != baseline[field]:
            problems.append(f"{field}: {baseline[field]} -> {current[field]}")

    base_ratios: dict[str, float] = dict(baseline.get("ball_ratios", {}))
    curr_ratios: dict[str, float] = dict(current.get("ball_ratios", {}))
    shared_keys = set(base_ratios) & set(curr_ratios)
    deviations = [
        (key, base_ratios[key], curr_ratios[key])
        for key in shared_keys
        if abs(base_ratios[key] - curr_ratios[key]) > BALL_RATIO_TOLERANCE
    ]
    if len(deviations) > MAX_DEVIATING_BALLS:
        worst = sorted(deviations, key=lambda d: -abs(d[1] - d[2]))[:5]
        details = ", ".join(f"{k}: {b} -> {c}" for k, b, c in worst)
        problems.append(
            f"{len(deviations)} balls deviate more than "
            f"{BALL_RATIO_TOLERANCE}pp (worst: {details})"
        )

    missing = set(base_ratios) - set(curr_ratios)
    if len(missing) > MAX_DEVIATING_BALLS:
        problems.append(f"{len(missing)} baseline pads no longer detected")

    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pipeline on reference images and compare metrics.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Write the current results as the new baseline.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="*",
        help="Explicit image paths (default: all Images_BGA/Raw images).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_PATH,
        help="Baseline JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = args.images if args.images else default_reference_images()
    if not images:
        print("[ERROR] No reference images found.")
        raise SystemExit(2)

    print(f"[INFO] Analyzing {len(images)} image(s)...")
    results: dict[str, dict[str, object]] = {}
    for image_path in images:
        started = time.perf_counter()
        results[image_path.name] = analyze_image(image_path)
        elapsed = time.perf_counter() - started
        summary = results[image_path.name]
        print(
            f"[INFO] {image_path.name}: balls={summary['balls_detected']} "
            f"avg={summary['average_void_ratio']}% "
            f"max={summary['maximum_void_ratio']}% "
            f"review={summary['balls_review']} fail={summary['balls_fail']} "
            f"({elapsed:.1f}s)"
        )

    if args.update or not args.baseline.exists():
        if not args.update:
            print("[INFO] No baseline found; creating one now.")
        args.baseline.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[OK] Baseline written: {args.baseline}")
        return

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    failed = False
    for name, current in results.items():
        if name not in baseline:
            print(f"[WARN] {name}: not in baseline (run with --update to add).")
            continue

        problems = compare_image_results(name, baseline[name], current)
        if problems:
            failed = True
            print(f"[FAIL] {name}:")
            for problem in problems:
                print(f"        - {problem}")
        else:
            print(f"[OK]   {name}")

    if failed:
        print("\n[RESULT] REGRESSION DETECTED - review the changes above.")
        raise SystemExit(1)

    print("\n[RESULT] All reference images match the baseline.")


if __name__ == "__main__":
    main()
