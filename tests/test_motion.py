"""Unit tests for motion matching and majority voting; no GPU required."""

import numpy as np
import pytest

from status_resolver.config import StatusConfig, CLS_VEHICLE, CLS_PERSON
from status_resolver.motion import MotionResolver, _greedy_match


# ---------------------------------------------------------------- _greedy_match

def test_greedy_match_exclusive():
    """Each source and destination may be matched at most once."""
    src = np.array([[0.0, 0.0], [10.0, 0.0]])
    dst = np.array([[1.0, 0.0], [11.0, 0.0]])
    cls = np.array([CLS_VEHICLE, CLS_VEHICLE], dtype=np.int32)
    radii = np.array([50.0, 50.0])

    m = _greedy_match(src, cls, dst, cls, radii)
    assert m == {0: 0, 1: 1}


def test_greedy_match_two_sources_one_target():
    """Only the closest source may claim a single destination."""
    src = np.array([[0.0, 0.0], [3.0, 0.0]])
    dst = np.array([[2.0, 0.0]])
    cls_src = np.array([CLS_VEHICLE, CLS_VEHICLE], dtype=np.int32)
    cls_dst = np.array([CLS_VEHICLE], dtype=np.int32)
    radii = np.array([50.0, 50.0])

    m = _greedy_match(src, cls_src, dst, cls_dst, radii)
    assert m == {1: 0}  # Distance 1 is closer than distance 2.


def test_greedy_match_class_compatibility():
    """Different classes never match."""
    src = np.array([[0.0, 0.0]])
    dst = np.array([[1.0, 0.0]])
    m = _greedy_match(
        src, np.array([CLS_VEHICLE], dtype=np.int32),
        dst, np.array([CLS_PERSON], dtype=np.int32),
        np.array([50.0]),
    )
    assert m == {}


def test_greedy_match_outside_radius():
    src = np.array([[0.0, 0.0]])
    dst = np.array([[100.0, 0.0]])
    cls = np.array([CLS_VEHICLE], dtype=np.int32)
    m = _greedy_match(src, cls, dst, cls, np.array([50.0]))
    assert m == {}


# ---------------------------------------------------------- majority vote

def _mk_resolver(**overrides):
    cfg = StatusConfig(**overrides)
    return MotionResolver(cfg)


def _smooth(resolver, raw, bbox=(100, 100, 300, 300), cls=CLS_VEHICLE):
    bboxes = np.array([bbox], dtype=np.float64)
    classes = np.array([cls], dtype=np.int32)
    statuses = np.array([raw], dtype=np.int8)
    return resolver._smooth_statuses(bboxes, classes, statuses)[0]


def test_young_track_is_stationary():
    """Young tracks stay stationary even with a moving raw estimate."""
    r = _mk_resolver(min_track_age=2)
    assert _smooth(r, 1) == 0  # Age 1 is below the minimum of 2.


def test_majority_vote_filters_jitter():
    """A single stationary vote does not override two moving votes."""
    r = _mk_resolver(min_track_age=1, smoothing_window=5)
    _smooth(r, 1)
    _smooth(r, 1)
    assert _smooth(r, 0) == 1  # Votes [1, 1, 0] have a moving majority.


def test_sustained_stop_detected():
    """Sustained stationary observations eventually change the output."""
    r = _mk_resolver(min_track_age=1, smoothing_window=3)
    _smooth(r, 1)
    _smooth(r, 0)
    _smooth(r, 0)
    assert _smooth(r, 0) == 0


def test_track_expires_after_gap():
    """Expired tracks are removed; returning detections start new tracks."""
    r = _mk_resolver(min_track_age=2, max_gap_frames=1, smoothing_window=5)
    _smooth(r, 1)  # Age 1.
    _smooth(r, 1)  # Age 2 now participates in voting.
    # Skip two frames to expire the track.
    empty_b = np.empty((0, 4)); empty_c = np.empty(0, dtype=np.int32); empty_s = np.empty(0, dtype=np.int8)
    r._smooth_statuses(empty_b, empty_c, empty_s)
    r._smooth_statuses(empty_b, empty_c, empty_s)
    assert len(r._tracks) == 0
    # A returning detection starts a young, stationary track.
    assert _smooth(r, 1) == 0


def test_empty_frames_advance_history():
    """Regression: empty frames advance history and track gaps to prevent stale references."""
    cfg = StatusConfig(min_track_age=1, max_gap_frames=10)
    r = MotionResolver(cfg)

    def frame(v):
        return np.full((80, 80, 3), v, dtype=np.uint8)

    bb = np.array([[15, 15, 40, 40]], dtype=np.float64)
    cl = np.array([CLS_VEHICLE], dtype=np.int32)
    empty_b = np.empty((0, 4), dtype=np.float64)
    empty_c = np.empty(0, dtype=np.int32)

    # Process three frames containing detections.
    for v in (10, 20, 30):
        r.update(bb, cl, frame(v), 80, 80)
    assert len(r._tracks) == 1
    assert r._tracks[0]["gap"] == 0

    # Advance three empty frames through the motion resolver.
    for v in (40, 50, 60):
        r.update(empty_b, empty_c, frame(v), 80, 80)

    # History must contain the most recent empty frames.
    history_grays = {int(h[0][0, 0]) for h in r._history}
    assert history_grays == {40, 50, 60}, f"Stale history: {history_grays}"

    # The track remains in memory and its gap has advanced.
    assert r._tracks[0]["gap"] == 3


def test_person_skipped_when_debug_disabled():
    """Persons are excluded from smoothing when debug motion is disabled."""
    r = _mk_resolver(debug_human_motion=False)
    result = _smooth(r, -1, cls=CLS_PERSON)
    assert result == -1
    assert len(r._tracks) == 0
