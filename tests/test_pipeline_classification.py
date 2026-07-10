"""Tests for the pipeline-facing classification and summary logic in main.py."""

import pytest

from src.main import classify_ball_quality, classify_image_quality
from src.reporting import build_summary

WARNING = 10.0
FAIL = 25.0
LARGEST_REVIEW = 12.25


def classify(total: float, largest: float) -> str:
    return classify_ball_quality(
        void_ratio_percent=total,
        largest_void_ratio_percent=largest,
        warning_threshold=WARNING,
        fail_threshold=FAIL,
        largest_void_fail_threshold=LARGEST_REVIEW,
    )


@pytest.mark.parametrize(
    ("total", "largest", "expected"),
    [
        (0.0, 0.0, "PASS"),
        (9.99, 12.24, "PASS"),
        (10.0, 0.0, "REVIEW"),
        (24.99, 0.0, "REVIEW"),
        (25.0, 0.0, "FAIL"),
        (5.0, 12.25, "REVIEW"),
        # Dominant void is a process indicator, never a defect (IPC-A-610
        # only rejects on cumulative area).
        (9.0, 24.9, "REVIEW"),
    ],
)
def test_classify_ball_quality_matches_ipc(total, largest, expected):
    assert classify(total, largest) == expected


def test_image_verdict_aggregation():
    assert classify_image_quality([{"quality": "PASS"}]) == "PASS"
    assert (
        classify_image_quality([{"quality": "PASS"}, {"quality": "REVIEW"}])
        == "REVIEW"
    )
    assert (
        classify_image_quality(
            [{"quality": "PASS"}, {"quality": "REVIEW"}, {"quality": "FAIL"}]
        )
        == "FAIL"
    )


def _row(total: float, largest: float) -> dict:
    return {
        "void_ratio_percent": total,
        "largest_void_ratio_percent": largest,
    }


def test_build_summary_basic_stats():
    rows = [_row(0.0, 0.0), _row(10.0, 6.0), _row(20.0, 8.0)]
    summary = build_summary("BOARD.jpg", rows, warning_threshold=WARNING)
    assert summary["total_balls"] == 3
    assert summary["average_void_ratio_percent"] == pytest.approx(10.0)
    assert summary["maximum_void_ratio_percent"] == pytest.approx(20.0)
    assert summary["max_single_void_ratio_percent"] == pytest.approx(8.0)
    assert summary["ipc_a610_defect_balls"] == 0
    assert summary["ipc_7095_process_indicator_balls"] == 2
    assert summary["ipc_verdict"] == "ACCEPT WITH PROCESS INDICATORS"


def test_build_summary_reject_and_empty():
    reject = build_summary("B.jpg", [_row(30.0, 25.0)], warning_threshold=WARNING)
    assert reject["ipc_verdict"] == "REJECT"
    assert reject["ipc_a610_defect_balls"] == 1

    empty = build_summary("B.jpg", [], warning_threshold=WARNING)
    assert empty["total_balls"] == 0
    assert empty["ipc_verdict"] == "NO BALLS DETECTED"
