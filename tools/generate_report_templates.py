"""Generate blank/sample report templates into the ``templates`` folder.

The templates are produced by the SAME reporting code that writes real
inspection reports, so they always stay in sync with the production format.
Sample data marks every value that a real run replaces.

Usage:
    python tools/generate_report_templates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import classify_ball_quality  # noqa: E402
from src.reporting import build_summary, save_reports  # noqa: E402

TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Representative sample balls: one per classification band so the template
# demonstrates every color/formatting rule of the real report.
SAMPLE_BALLS = [
    # (ball_id, x, y, void%, largest%, estimated)
    (1, 500, 400, 0.00, 0.00, False),
    (2, 620, 400, 1.85, 1.20, False),
    (3, 740, 400, 4.50, 2.10, False),
    (4, 860, 400, 8.75, 5.40, False),
    (5, 980, 400, 13.20, 7.90, False),   # REVIEW: 10-25% band
    (6, 1100, 400, 18.60, 14.10, False), # REVIEW: dominant single void
    (7, 1220, 400, 27.40, 21.00, False), # FAIL: above IPC-A-610 25%
    (8, 1340, 400, 2.30, 1.10, True),    # grid-estimated pad
]

WARNING_THRESHOLD = 10.0
FAIL_THRESHOLD = 25.0
LARGEST_VOID_REVIEW_THRESHOLD = 12.25


def build_sample_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ball_area = 2827.43  # radius 30 px
    for ball_id, x, y, void_pct, largest_pct, estimated in SAMPLE_BALLS:
        quality = classify_ball_quality(
            void_ratio_percent=void_pct,
            largest_void_ratio_percent=largest_pct,
            warning_threshold=WARNING_THRESHOLD,
            fail_threshold=FAIL_THRESHOLD,
            largest_void_fail_threshold=LARGEST_VOID_REVIEW_THRESHOLD,
        )
        rows.append(
            {
                "image_name": "PCBA_TEMPLATE.jpg",
                "ball_id": ball_id,
                "center_x": x,
                "center_y": y,
                "diameter_px": 60,
                "ball_area_px": ball_area,
                "void_count": 2 if void_pct else 0,
                "total_void_area_px": round(ball_area * void_pct / 100.0),
                "largest_void_area_px": round(ball_area * largest_pct / 100.0),
                "void_ratio_percent": void_pct,
                "largest_void_ratio_percent": largest_pct,
                "quality": quality,
                "is_estimated_pad": estimated,
                "confidence": 0.0 if estimated else 0.95,
            }
        )
    return rows


def main() -> None:
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_sample_rows()
    summary = build_summary(
        image_name="PCBA_TEMPLATE.jpg",
        rows=rows,
        warning_threshold=WARNING_THRESHOLD,
    )
    context = {
        "image_path": "Images_BGA/Raw/PCBA_TEMPLATE.jpg",
        "image_width": 2436,
        "image_height": 2042,
        "expected_slots": 256,
        "occupied_slots": 256,
        "warning_threshold": WARNING_THRESHOLD,
        "fail_threshold": FAIL_THRESHOLD,
        "largest_void_review_threshold": LARGEST_VOID_REVIEW_THRESHOLD,
        "preview_image": None,
        "processing_seconds": 0.0,
    }

    paths = save_reports(
        rows=rows,
        summary=summary,
        output_dir=TEMPLATES_DIR,
        image_stem="BGA_Report_TEMPLATE",
        context=context,
    )

    for kind, path in paths.items():
        print(f"[OK] {kind}: {path}")
    print(
        "\n[INFO] Templates use 8 sample balls covering every band "
        "(acceptable / process indicator / defect / grid-estimated) so all "
        "formatting rules of the production report are demonstrated."
    )


if __name__ == "__main__":
    main()
