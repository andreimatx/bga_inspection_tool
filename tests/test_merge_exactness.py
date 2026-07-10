"""Equivalence test for the spatial-hash candidate merger.

``_merge_pad_candidates`` was rewritten from an O(n^2) scan to a
spatial-hash search for speed. This test pins the guarantee that motivated
the rewrite: on randomized inputs the fast version must return EXACTLY the
same candidates, in the same order, as the original brute-force algorithm
(reimplemented below as the reference).
"""

import random

import numpy as np

from src.ball_detection import HoughCircleConfig, _merge_pad_candidates, _PadCandidate


def _reference_merge(candidates, config):
    """The original O(n^2) implementation, kept verbatim as ground truth."""
    merged = []
    ordered = sorted(
        candidates,
        key=lambda item: (item.score, item.radius),
        reverse=True,
    )
    for candidate in ordered:
        duplicate_index = None
        for index, existing in enumerate(merged):
            distance = float(
                np.hypot(candidate.x - existing.x, candidate.y - existing.y)
            )
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


def _random_candidates(rng, count, span=2000):
    return [
        _PadCandidate(
            x=rng.randint(0, span),
            y=rng.randint(0, span),
            radius=rng.randint(8, 60),
            score=round(rng.uniform(0.0, 1.0), 4),
            source=rng.choice(["hough", "blob"]),
        )
        for _ in range(count)
    ]


def test_merge_matches_bruteforce_on_random_inputs():
    config = HoughCircleConfig()
    rng = random.Random(1234)
    for trial in range(20):
        # Mix of sparse candidates and dense clusters (many duplicates).
        candidates = _random_candidates(rng, 300, span=2000)
        candidates += _random_candidates(rng, 100, span=200)
        assert _merge_pad_candidates(candidates, config) == _reference_merge(
            candidates, config
        ), f"divergence at trial {trial}"


def test_merge_empty_and_single():
    config = HoughCircleConfig()
    assert _merge_pad_candidates([], config) == []
    one = [_PadCandidate(x=10, y=10, radius=20, score=0.5, source="hough")]
    assert _merge_pad_candidates(one, config) == one


def test_merge_keeps_strongest_of_duplicates():
    config = HoughCircleConfig()
    weak = _PadCandidate(x=100, y=100, radius=20, score=0.4, source="hough")
    strong = _PadCandidate(x=102, y=101, radius=20, score=0.9, source="blob")
    far = _PadCandidate(x=400, y=400, radius=20, score=0.1, source="hough")
    merged = _merge_pad_candidates([weak, strong, far], config)
    assert strong in merged
    assert weak not in merged
    assert far in merged
