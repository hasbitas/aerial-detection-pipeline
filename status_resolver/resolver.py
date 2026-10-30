"""
StatusResolver — main orchestrator.
Compatible with standard YOLO predict results, no tracking required.
Now exposes per-stage timing for profiling.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .config import StatusConfig, MOTION_CLASSES
from .landing import LandingResolver
from .motion import MotionResolver


class StatusResolver:

    def __init__(self, config: StatusConfig | None = None):
        self.config = config or StatusConfig()
        self.landing = LandingResolver(self.config)
        self.motion = MotionResolver(self.config)

        # Per-call timing (set after each update() call)
        self.last_timing = {
            "extract_ms": 0.0,
            "egomotion_ms": 0.0,
            "landing_ms": 0.0,
            "assemble_ms": 0.0,
        }

    def update(
        self,
        frame: np.ndarray,
        results: Any,
        class_map: dict[int, int] | None = None,
    ) -> list[dict]:
        """Process one frame. class_map maps model class IDs to canonical output IDs."""
        # ---- STAGE: bbox extraction (GPU→CPU transfer + numpy convert) ----
        t0 = time.perf_counter()

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            # Advance motion history and track ages even on empty frames.
            # Stale references after gaps can produce false motion.
            # Stationary vehicles must not inherit stale references.
            t_empty_0 = time.perf_counter()
            img_h, img_w = frame.shape[:2]
            self.motion.update(
                np.empty((0, 4), dtype=np.float64),
                np.empty(0, dtype=np.int32),
                frame, img_h, img_w,
            )
            t_empty_1 = time.perf_counter()
            self.last_timing = {k: 0.0 for k in self.last_timing}
            self.last_timing["egomotion_ms"] = (t_empty_1 - t_empty_0) * 1000
            return []

        bboxes = boxes.xyxy.cpu().numpy().astype(np.float64)
        classes = boxes.cls.cpu().numpy().astype(np.int32)
        if class_map:
            classes = np.array(
                [class_map.get(int(c), int(c)) for c in classes], dtype=np.int32
            )
        confs = boxes.conf.cpu().numpy().astype(np.float64)
        img_h, img_w = frame.shape[:2]

        t1 = time.perf_counter()

        # ---- STAGE: ego-motion + motion resolve ----
        motion_statuses = self.motion.update(
            bboxes, classes, frame, img_h, img_w
        )
        t2 = time.perf_counter()

        # ---- STAGE: landing resolve ----
        landing_statuses = self.landing.resolve_batch(
            bboxes, classes, img_h, img_w
        )
        t3 = time.perf_counter()

        # ---- STAGE: assemble output dicts ----
        output = []
        for i in range(len(classes)):
            raw_motion = int(motion_statuses[i])
            # Only vehicles receive official motion codes 0/1; others receive -1.
            # The debug field preserves the raw estimate.
            # Optional person motion is available only in the debug field.
            official_motion = raw_motion if int(classes[i]) in MOTION_CLASSES else -1
            output.append({
                "bbox": bboxes[i].tolist(),
                "cls": int(classes[i]),
                "conf": float(confs[i]),
                "track_id": -1,
                "motion_status": official_motion,
                "motion_status_debug": raw_motion,
                "landing_status": int(landing_statuses[i]),
            })
        t4 = time.perf_counter()

        self.last_timing = {
            "extract_ms": (t1 - t0) * 1000,
            "egomotion_ms": (t2 - t1) * 1000,
            "landing_ms": (t3 - t2) * 1000,
            "assemble_ms": (t4 - t3) * 1000,
        }

        return output
