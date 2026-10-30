"""
Landing status resolver.
Stateless — works on a single frame, no history needed.

Rules (applied in order):
1. Edge check: UAP/UAI bbox near image border → 0 (area clipped)
2. Obstacle overlap: UAP/UAI overlaps with obstacle (human/vehicle) → 0
3. Otherwise → 1 (suitable)

Non-landing classes (Vehicle, Person) → -1
"""

from __future__ import annotations

import numpy as np

from .config import StatusConfig, LANDING_CLASSES, OBSTACLE_CLASSES


class LandingResolver:

    def __init__(self, config: StatusConfig):
        self.config = config

    def resolve(
        self,
        bbox: np.ndarray,
        cls: int,
        all_bboxes: np.ndarray,
        all_classes: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> int:
        """
        Determine landing status for a single detection.

        Args:
            bbox: [x1, y1, x2, y2] of the target UAP/UAI.
            cls: class index of the target detection.
            all_bboxes: (N, 4) array of all detections in this frame.
            all_classes: (N,) array of class indices for all detections.
            img_h: frame height in pixels.
            img_w: frame width in pixels.

        Returns:
            1  = suitable for landing
            0  = not suitable
            -1 = not a landing area class
        """
        if cls not in LANDING_CLASSES:
            return -1

        # Rule 1: edge check
        if self._is_clipped(bbox, img_h, img_w):
            return 0

        # Rule 2: obstacle overlap (human / vehicle only)
        for i in range(len(all_classes)):
            if all_classes[i] not in OBSTACLE_CLASSES:
                continue
            other_bbox = all_bboxes[i]
            if self._has_containment_overlap(bbox, other_bbox):
                return 0

        return 1

    def resolve_batch(
        self,
        bboxes: np.ndarray,
        classes: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> np.ndarray:
        """
        Resolve landing status for all detections in a frame.

        Args:
            bboxes: (N, 4) all detection bboxes [x1, y1, x2, y2].
            classes: (N,) class indices.
            img_h, img_w: frame dimensions.

        Returns:
            (N,) int array of landing statuses.
        """
        n = len(classes)
        statuses = np.full(n, -1, dtype=np.int8)
        if n == 0:
            return statuses

        landing_mask = np.isin(classes, list(LANDING_CLASSES))
        if not landing_mask.any():
            return statuses

        # Rule 1 (vectorized): edge-clipped areas are unsuitable.
        eps = self.config.edge_margin_px
        clipped = (
            (bboxes[:, 0] < eps) | (bboxes[:, 1] < eps)
            | (bboxes[:, 2] > img_w - eps) | (bboxes[:, 3] > img_h - eps)
        )
        statuses[landing_mask] = 1
        statuses[landing_mask & clipped] = 0

        # Rule 2 (vectorized): containment above threshold blocks landing.
        pad_idx = np.where(landing_mask & ~clipped)[0]
        obs_idx = np.where(np.isin(classes, list(OBSTACLE_CLASSES)))[0]
        if len(pad_idx) > 0 and len(obs_idx) > 0:
            p = bboxes[pad_idx]   # (L, 4)
            o = bboxes[obs_idx]   # (O, 4)

            ix1 = np.maximum(p[:, None, 0], o[None, :, 0])
            iy1 = np.maximum(p[:, None, 1], o[None, :, 1])
            ix2 = np.minimum(p[:, None, 2], o[None, :, 2])
            iy2 = np.minimum(p[:, None, 3], o[None, :, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)  # (L, O)

            area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])  # (L,)
            area_o = (o[:, 2] - o[:, 0]) * (o[:, 3] - o[:, 1])  # (O,)
            min_area = np.minimum(area_p[:, None], area_o[None, :])

            with np.errstate(divide="ignore", invalid="ignore"):
                containment = np.where(min_area > 0, inter / min_area, 0.0)

            blocked = (containment > self.config.containment_thresh).any(axis=1)
            statuses[pad_idx[blocked]] = 0

        return statuses

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_clipped(self, bbox: np.ndarray, img_h: int, img_w: int) -> bool:
        """True if bbox is near image boundary (area likely clipped)."""
        x1, y1, x2, y2 = bbox
        eps = self.config.edge_margin_px
        return x1 < eps or y1 < eps or x2 > img_w - eps or y2 > img_h - eps

    def _has_containment_overlap(
        self, pad_bbox: np.ndarray, obstacle_bbox: np.ndarray
    ) -> bool:
        """
        Containment-based overlap check.
        Uses intersection / min(area_pad, area_obstacle).

        Why not IoU: a small human (2000px²) fully inside a large pad (40000px²)
        gives IoU = 2000/40000 = 0.05 — effectively invisible to IoU.
        Containment metric gives 2000/2000 = 1.0 — correctly detected.
        """
        # Intersection
        ix1 = max(pad_bbox[0], obstacle_bbox[0])
        iy1 = max(pad_bbox[1], obstacle_bbox[1])
        ix2 = min(pad_bbox[2], obstacle_bbox[2])
        iy2 = min(pad_bbox[3], obstacle_bbox[3])

        if ix2 <= ix1 or iy2 <= iy1:
            return False  # No intersection at all

        intersection = (ix2 - ix1) * (iy2 - iy1)

        area_pad = (pad_bbox[2] - pad_bbox[0]) * (pad_bbox[3] - pad_bbox[1])
        area_obs = (obstacle_bbox[2] - obstacle_bbox[0]) * (obstacle_bbox[3] - obstacle_bbox[1])

        if area_pad <= 0 or area_obs <= 0:
            return False

        containment = intersection / min(area_pad, area_obs)
        return containment > self.config.containment_thresh
