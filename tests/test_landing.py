"""Unit tests for landing suitability rules."""

import numpy as np
import pytest

from status_resolver.config import StatusConfig, CLS_VEHICLE, CLS_PERSON, CLS_UAP, CLS_UAI
from status_resolver.landing import LandingResolver

IMG_W, IMG_H = 1920, 1080


@pytest.fixture
def resolver():
    return LandingResolver(StatusConfig())


def _batch(resolver, bboxes, classes):
    return resolver.resolve_batch(
        np.array(bboxes, dtype=np.float64),
        np.array(classes, dtype=np.int32),
        IMG_H, IMG_W,
    )


def test_clear_area_is_suitable(resolver):
    """A fully visible area without obstacles is suitable."""
    statuses = _batch(resolver, [[500, 300, 900, 700]], [CLS_UAP])
    assert statuses[0] == 1


def test_edge_clipped_area_is_unsuitable(resolver):
    """An area clipped at the image edge is unsuitable."""
    statuses = _batch(resolver, [[0, 300, 400, 700]], [CLS_UAI])
    assert statuses[0] == 0


def test_person_blocks_landing(resolver):
    """A person fully inside the area has containment 1.0 and blocks landing."""
    statuses = _batch(
        resolver,
        [[500, 300, 900, 700], [650, 450, 700, 550]],
        [CLS_UAP, CLS_PERSON],
    )
    assert statuses[0] == 0
    assert statuses[1] == -1  # A person is not a landing area.


def test_overlap_below_threshold_is_suitable(resolver):
    """Overlap below the containment threshold does not block landing."""
    # Obstacle area 10000, intersection 400: containment 0.04.
    statuses = _batch(
        resolver,
        [[500, 300, 900, 700], [880, 680, 980, 780]],
        [CLS_UAP, CLS_VEHICLE],
    )
    assert statuses[0] == 1


def test_vehicles_and_persons_are_not_landing_areas(resolver):
    statuses = _batch(
        resolver,
        [[100, 100, 200, 200], [300, 300, 350, 400]],
        [CLS_VEHICLE, CLS_PERSON],
    )
    assert list(statuses) == [-1, -1]


def test_batch_matches_scalar(resolver):
    """Vectorized and scalar resolution must agree."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = int(rng.integers(1, 12))
        x1 = rng.uniform(0, IMG_W - 100, n)
        y1 = rng.uniform(0, IMG_H - 100, n)
        w = rng.uniform(20, 500, n)
        h = rng.uniform(20, 500, n)
        bboxes = np.stack([x1, y1, np.minimum(x1 + w, IMG_W), np.minimum(y1 + h, IMG_H)], axis=1)
        classes = rng.integers(0, 4, n).astype(np.int32)

        batch = resolver.resolve_batch(bboxes, classes, IMG_H, IMG_W)
        scalar = [
            resolver.resolve(bboxes[i], int(classes[i]), bboxes, classes, IMG_H, IMG_W)
            for i in range(n)
        ]
        assert list(batch) == scalar
