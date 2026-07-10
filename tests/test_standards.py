"""Boundary tests for the IPC-A-610 / IPC-7095 classification logic.

These thresholds decide PASS / REVIEW / FAIL for every ball and board, so
they are pinned here at their exact boundaries: a silent change to any of
them shows up in milliseconds instead of a minutes-long image regression.
"""

import pytest

from src.standards import (
    IPCVoidCriteria,
    bucket_void_distribution,
    classify_ball_ipc,
    evaluate_board,
    ipc_assessment_label,
    void_diameter_ratio_percent,
)


# ---------------------------------------------------------------------------
# Per-ball classification (classify_ball_ipc)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("total", "largest", "expected"),
    [
        # Clean balls.
        (0.0, 0.0, "PASS"),
        (9.99, 5.0, "PASS"),
        # IPC-7095 process-indicator band starts at exactly 10%.
        (10.0, 5.0, "REVIEW"),
        (24.99, 5.0, "REVIEW"),
        # IPC-A-610 defect starts at exactly 25% cumulative area.
        (25.0, 5.0, "FAIL"),
        (99.0, 99.0, "FAIL"),
        # Dominant single void: REVIEW at exactly 12.25% area (~35% dia).
        (5.0, 12.24, "PASS"),
        (5.0, 12.25, "REVIEW"),
        # A dominant void alone must NEVER escalate to FAIL.
        (9.0, 24.0, "REVIEW"),
    ],
)
def test_classify_ball_ipc_boundaries(total, largest, expected):
    assert classify_ball_ipc(total, largest) == expected


def test_classify_ball_defect_wins_over_indicator():
    # 30% total is a defect even with a tiny largest void.
    assert classify_ball_ipc(30.0, 1.0) == "FAIL"


def test_ipc_assessment_labels():
    assert ipc_assessment_label("PASS") == "ACCEPTABLE"
    assert ipc_assessment_label("REVIEW") == "PROCESS INDICATOR"
    assert ipc_assessment_label("FAIL") == "DEFECT (IPC-A-610)"


# ---------------------------------------------------------------------------
# Diameter <-> area conversion (IPC-7095 wording)
# ---------------------------------------------------------------------------
def test_void_diameter_ratio_50_percent_diameter_is_25_percent_area():
    # A void with 25% of the ball AREA has 50% of the ball DIAMETER.
    assert void_diameter_ratio_percent(25.0, 100.0) == pytest.approx(50.0)


def test_void_diameter_ratio_handles_zero():
    assert void_diameter_ratio_percent(0.0, 100.0) == 0.0
    assert void_diameter_ratio_percent(10.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Distribution buckets
# ---------------------------------------------------------------------------
def test_bucket_distribution_counts_every_ball_once():
    ratios = [0.0, 1.9, 2.0, 4.99, 5.0, 9.99, 10.0, 14.9, 15.0, 24.9, 25.0, 80.0]
    distribution = bucket_void_distribution(ratios)
    assert sum(count for _, count in distribution) == len(ratios)
    counts = dict(distribution)
    assert counts["0 - 2%"] == 2       # 0.0, 1.9
    assert counts["2 - 5%"] == 2       # 2.0, 4.99
    assert counts["5 - 10%"] == 2      # 5.0, 9.99
    assert counts["10 - 15%"] == 2     # 10.0, 14.9
    assert counts["15 - 25%"] == 2     # 15.0, 24.9
    assert counts["> 25%"] == 2        # 25.0, 80.0


# ---------------------------------------------------------------------------
# Board-level evaluation
# ---------------------------------------------------------------------------
def _ball(total: float, largest: float) -> dict:
    return {
        "void_ratio_percent": total,
        "largest_void_ratio_percent": largest,
    }


def test_board_accept_when_all_clean():
    board = evaluate_board([_ball(1.0, 0.5), _ball(9.9, 4.0)])
    assert board["ipc_verdict"] == "ACCEPT"
    assert board["defect_balls"] == 0
    assert board["process_indicator_balls"] == 0


def test_board_process_indicators_in_band():
    board = evaluate_board([_ball(1.0, 0.5), _ball(15.0, 8.0)])
    assert board["ipc_verdict"] == "ACCEPT WITH PROCESS INDICATORS"
    assert board["process_indicator_balls"] == 1
    assert board["defect_balls"] == 0


def test_board_process_indicator_from_dominant_void_only():
    # Total below 10% but one dominant void still flags the board.
    board = evaluate_board([_ball(9.0, 13.0)])
    assert board["ipc_verdict"] == "ACCEPT WITH PROCESS INDICATORS"
    assert board["dominant_void_balls"] == 1


def test_board_reject_on_single_defect_ball():
    board = evaluate_board([_ball(1.0, 0.5), _ball(25.0, 20.0)])
    assert board["ipc_verdict"] == "REJECT"
    assert board["defect_balls"] == 1


def test_criteria_defaults_match_ipc():
    criteria = IPCVoidCriteria()
    assert criteria.defect_area_percent == 25.0
    assert criteria.process_indicator_area_percent == 10.0
    assert criteria.largest_void_review_area_percent == 12.25
