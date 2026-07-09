"""IPC-based acceptance criteria for BGA void inspection.

The numbers in this module implement the two industry references that govern
void assessment in BGA solder joints on 2D X-ray images:

- IPC-A-610 (Acceptability of Electronic Assemblies), BGA section:
  a solder ball is a DEFECT for Class 1, 2 and 3 products when the cumulative
  projected void area exceeds 25% of the ball area in the X-ray image.
- IPC-7095 (Design and Assembly Process Implementation for BGAs):
  process-level guidance. Voiding between 10% and 25% is not a defect but a
  process indicator worth monitoring; voids at the pad interface and single
  large voids (diameter above ~50% of the ball diameter, i.e. 25% of the
  area) deserve engineering attention. IPC-7095 also classifies void types
  (macrovoids, planar microvoids, shrinkage voids, via-in-pad voids).

IMPORTANT 2D LIMITATION: a single top-down X-ray cannot separate voids at the
package interface, mid-ball and pad interface: everything is projected onto
one plane. This tool therefore measures the PROJECTED cumulative void area
per ball and applies the 25% projected-area criterion, which is exactly how
IPC-A-610 phrases the X-ray requirement. Location-specific IPC-7095 limits
(e.g. 10% at the pad interface) cannot be verified from one view and are
reported as "not evaluated" rather than silently guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class IPCVoidCriteria:
    """Thresholds used for report classification, with their IPC sources."""

    # IPC-A-610, BGA voiding: DEFECT above 25% cumulative projected area
    # (Class 1, 2 and 3 share the same X-ray criterion).
    defect_area_percent: float = 25.0

    # IPC-7095 process guidance: voiding above ~10% is worth monitoring even
    # though it is acceptable per IPC-A-610. Also matches common Class 3
    # workmanship expectations at interconnect interfaces.
    process_indicator_area_percent: float = 10.0

    # A single void whose diameter exceeds ~35% of the ball diameter
    # (~12.25% of the projected area) is flagged for engineering review:
    # IPC-7095 treats large single voids as more serious than the same area
    # spread across many small voids, because one large void concentrates
    # stress and can sit at an interface.
    largest_void_review_area_percent: float = 12.25

    standard_names: str = "IPC-A-610 / IPC-7095"


# Distribution buckets used in the report histogram (upper bounds, percent).
VOID_BUCKETS: list[tuple[str, float, float]] = [
    ("0 - 2%", 0.0, 2.0),
    ("2 - 5%", 2.0, 5.0),
    ("5 - 10%", 5.0, 10.0),
    ("10 - 15%", 10.0, 15.0),
    ("15 - 25%", 15.0, 25.0),
    ("> 25%", 25.0, float("inf")),
]


def classify_ball_ipc(
    void_ratio_percent: float,
    largest_void_ratio_percent: float,
    criteria: IPCVoidCriteria = IPCVoidCriteria(),
) -> str:
    """Classify one ball: PASS / REVIEW (process indicator) / FAIL (defect).

    FAIL maps to the IPC-A-610 DEFECT condition (>25% cumulative projected
    area). REVIEW is a process indicator per IPC-7095 guidance: either total
    voiding in the 10-25% band, or one single dominant void.
    """
    if void_ratio_percent >= criteria.defect_area_percent:
        return "FAIL"

    if void_ratio_percent >= criteria.process_indicator_area_percent:
        return "REVIEW"

    if largest_void_ratio_percent >= criteria.largest_void_review_area_percent:
        return "REVIEW"

    return "PASS"


def ipc_assessment_label(quality: str) -> str:
    """Map the internal quality code to the wording used in reports."""
    return {
        "PASS": "ACCEPTABLE",
        "REVIEW": "PROCESS INDICATOR",
        "FAIL": "DEFECT (IPC-A-610)",
    }.get(quality, quality)


def void_diameter_ratio_percent(
    void_area_px: float,
    ball_area_px: float,
) -> float:
    """Equivalent void diameter as percent of the ball diameter.

    IPC-7095 sometimes expresses limits in diameter terms (a void of 50% of
    the ball diameter equals 25% of the projected area). Assumes both the
    ball and the void are near-circular, which matches how this pipeline
    renders voids (fitted circles).
    """
    if ball_area_px <= 0 or void_area_px <= 0:
        return 0.0
    return sqrt(void_area_px / ball_area_px) * 100.0


def bucket_void_distribution(ratios: list[float]) -> list[tuple[str, int]]:
    """Count balls per void-ratio bucket for the report histogram."""
    counts: list[tuple[str, int]] = []
    for label, low, high in VOID_BUCKETS:
        count = sum(1 for ratio in ratios if low <= ratio < high)
        counts.append((label, count))
    return counts


def evaluate_board(
    rows: list[dict[str, object]],
    criteria: IPCVoidCriteria = IPCVoidCriteria(),
) -> dict[str, object]:
    """Board-level IPC evaluation used by the Excel/Word reports."""
    ratios = [float(row["void_ratio_percent"]) for row in rows]
    largest_ratios = [
        float(row.get("largest_void_ratio_percent", 0.0)) for row in rows
    ]

    defect_balls = sum(
        ratio >= criteria.defect_area_percent for ratio in ratios
    )
    indicator_balls = sum(
        criteria.process_indicator_area_percent
        <= ratio
        < criteria.defect_area_percent
        for ratio in ratios
    )
    dominant_void_balls = sum(
        largest >= criteria.largest_void_review_area_percent
        for largest in largest_ratios
    )

    if defect_balls > 0:
        verdict = "REJECT"
        verdict_detail = (
            f"{defect_balls} ball(s) exceed the IPC-A-610 25% cumulative "
            "void area limit."
        )
    elif indicator_balls > 0 or dominant_void_balls > 0:
        verdict = "ACCEPT WITH PROCESS INDICATORS"
        verdict_detail = (
            "No IPC-A-610 defect. "
            f"{indicator_balls} ball(s) in the 10-25% process-indicator band"
            + (
                f"; {dominant_void_balls} ball(s) carry one dominant void"
                if dominant_void_balls
                else ""
            )
            + " (IPC-7095 process guidance)."
        )
    else:
        verdict = "ACCEPT"
        verdict_detail = (
            "All balls below the 10% process-indicator level; well within "
            "the IPC-A-610 25% acceptance limit."
        )

    max_largest = max(largest_ratios) if largest_ratios else 0.0

    return {
        "ipc_verdict": verdict,
        "ipc_verdict_detail": verdict_detail,
        "defect_balls": defect_balls,
        "process_indicator_balls": indicator_balls,
        "dominant_void_balls": dominant_void_balls,
        "max_total_void_percent": max(ratios) if ratios else 0.0,
        "max_single_void_percent": max_largest,
        "max_single_void_diameter_percent": (
            sqrt(max_largest / 100.0) * 100.0 if max_largest > 0 else 0.0
        ),
        "criteria": criteria,
        "distribution": bucket_void_distribution(ratios),
    }
