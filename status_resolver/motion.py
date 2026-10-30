"""
Motion status resolver without tracking IDs.
Uses optical flow for ego-motion compensation.
Compares current frame to N frames back (ring buffer) to smooth out jitter.
Supports resolution downscaling for optimized performance.
"""

from __future__ import annotations

import numpy as np
import cv2
from collections import deque

from .config import StatusConfig, MOTION_CLASSES, CLS_PERSON


class MotionResolver:

    def __init__(self, config: StatusConfig):
        self.config = config
        # Ring buffer: each element is a (gray_ds, bboxes, classes, centers, features) tuple
        self._history = deque(maxlen=config.motion_frame_skip)
        # Lightweight tracks preserve identity for majority voting.
        # Each entry: {"center", "cls", "votes", "age", "gap"}
        self._tracks: list[dict] = []
        # Compute motion for vehicles and optionally persons in debug mode.
        self._motion_set = frozenset(
            MOTION_CLASSES | ({CLS_PERSON} if config.debug_human_motion else set())
        )

    def update(
        self,
        bboxes: np.ndarray,
        classes: np.ndarray,
        frame: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> np.ndarray:
        """
        Determine motion status by comparing current detections
        to detections from N frames back (motion_frame_skip).
        """
        n = len(classes)
        statuses = np.full(n, -1, dtype=np.int8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Downscale gray frame for ego-motion and optical flow tracking.
        # Adaptive mode targets roughly 270 pixels in height.
        # Use a factor of 4 at Full HD and 8 at 4K.
        if self.config.adaptive_downscale:
            ds = max(1, round(img_h / 270))
        else:
            ds = self.config.downscale_factor
        if ds > 1:
            gray_ds = cv2.resize(gray, (0, 0), fx=1.0/ds, fy=1.0/ds, interpolation=cv2.INTER_LINEAR)
        else:
            gray_ds = gray

        # First frames (until history is full) -> all stationary (0)
        if len(self._history) < self.config.motion_frame_skip:
            for i in range(n):
                if classes[i] in self._motion_set:
                    statuses[i] = 0
            statuses = self._smooth_statuses(bboxes, classes, statuses)
            self._push_state(gray_ds, bboxes, classes, ds)
            return statuses

        # Retrieve reference state from N frames back
        ref_gray_ds, ref_bboxes, ref_classes, ref_centers, ref_features = self._history[0]

        # 1. Ego-motion: Estimate warp matrix between ref_gray_ds and gray_ds
        warp_matrix = np.eye(2, 3, dtype=np.float64)
        warp_valid = False
        noise_px = 0.0  # Residual noise after compensation, in original-image pixels.

        if ref_features is not None and len(ref_features) > 10:
            p1, st, err = cv2.calcOpticalFlowPyrLK(ref_gray_ds, gray_ds, ref_features, None)
            good_new = p1[st == 1]
            good_old = ref_features[st == 1]

            if len(good_new) >= 4:
                # estimateAffinePartial2D on downscaled features
                M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
                if M is not None:
                    # Scale warp matrix back to original coordinate system (rotation remains same, translation scales by ds)
                    warp_matrix = M.copy()
                    warp_matrix[0, 2] *= ds
                    warp_matrix[1, 2] *= ds
                    warp_valid = True

                    # Estimate adaptive noise from unexplained warp residuals.
                    # Background points estimate the current jitter level.
                    pred = good_old @ M[:2, :2].T + M[:, 2]
                    residuals = np.linalg.norm(pred - good_new, axis=1)
                    if inliers is not None and inliers.any():
                        noise_px = float(np.median(residuals[inliers.ravel().astype(bool)])) * ds
                    else:
                        noise_px = float(np.median(residuals)) * ds

        # 2. Track vehicle centers using Optical Flow on downscaled images.
        # Fall back to the reference center when optical flow tracking fails.
        # Invalid coordinates must not affect matching.
        pred_centers = _bbox_centers(ref_bboxes)
        if ref_centers is not None and len(ref_centers) > 0:
            ref_centers_ds = (ref_centers / ds).astype(np.float32).reshape(-1, 1, 2)
            p1_centers, st_centers, _ = cv2.calcOpticalFlowPyrLK(
                ref_gray_ds, gray_ds,
                ref_centers_ds,
                None
            )
            if p1_centers is not None:
                tracked = p1_centers.reshape(-1, 2) * ds
                ok = st_centers.reshape(-1).astype(bool)
                pred_centers[ok] = tracked[ok]

        # Match and classify with exclusive global greedy matching.
        curr_centers = _bbox_centers(bboxes)
        curr_diags = _bbox_diagonals(bboxes)

        # Evaluate motion classes whose boxes are not edge-clipped.
        eligible = []
        for i in range(n):
            if classes[i] not in self._motion_set:
                continue
            if self._is_edge_clipped(bboxes[i], img_h, img_w):
                statuses[i] = 0
                continue
            eligible.append(i)
            statuses[i] = 0  # Default to stationary when unmatched.

        if eligible:
            src_idx = np.array(eligible, dtype=np.int64)
            matches = _greedy_match(
                curr_centers[src_idx], classes[src_idx],
                pred_centers, ref_classes,
                curr_diags[src_idx] * self.config.match_radius_diag,
            )

            adaptive_floor_px = max(
                self.config.min_motion_px,
                self.config.noise_gain * noise_px,
            )

            for s, j in matches.items():
                i = int(src_idx[s])
                ref_center = _bbox_center(ref_bboxes[j])
                comp_ref_center = _warp_point(ref_center, warp_matrix) if warp_valid else ref_center

                displacement = float(np.linalg.norm(curr_centers[i] - comp_ref_center))
                curr_diag = curr_diags[i]
                normalized_res = displacement / curr_diag if curr_diag > 0 else 0

                thresh = (self.config.human_residual_thresh
                          if classes[i] == CLS_PERSON
                          else self.config.normalized_residual_thresh)

                if normalized_res > thresh and displacement > adaptive_floor_px:
                    statuses[i] = 1

        # Majority voting filters single-frame jitter.
        statuses = self._smooth_statuses(bboxes, classes, statuses)

        self._push_state(gray_ds, bboxes, classes, ds)
        return statuses

    def _smooth_statuses(
        self,
        bboxes: np.ndarray,
        classes: np.ndarray,
        raw_statuses: np.ndarray,
    ) -> np.ndarray:
        """Lightweight same-class track matching and majority voting. Match within the configured diagonal radius; young tracks default to stationary."""
        n = len(classes)
        final = raw_statuses.copy()

        # Detections eligible for evaluation.
        eligible = [
            i for i in range(n)
            if classes[i] in self._motion_set and raw_statuses[i] != -1
        ]
        if not eligible and not self._tracks:
            return final

        src_idx = np.array(eligible, dtype=np.int64)
        matched_tracks: set[int] = set()
        det_to_track: dict[int, int] = {}

        if eligible and self._tracks:
            centers = _bbox_centers(bboxes)
            diags = _bbox_diagonals(bboxes)
            track_centers = np.array([tr["center"] for tr in self._tracks])
            track_classes = np.array([tr["cls"] for tr in self._tracks], dtype=np.int32)
            matches = _greedy_match(
                centers[src_idx], classes[src_idx],
                track_centers, track_classes,
                diags[src_idx] * self.config.match_radius_diag,
            )
            det_to_track = {int(src_idx[s]): t for s, t in matches.items()}
            matched_tracks = set(det_to_track.values())

        # Track indices updated in this frame, including new tracks.
        updated_ids: set[int] = set(matched_tracks)

        for i in eligible:
            if i in det_to_track:
                tr = self._tracks[det_to_track[i]]
                tr["center"] = _bbox_center(bboxes[i])
                tr["votes"].append(int(raw_statuses[i]))
                tr["age"] += 1
                tr["gap"] = 0
            else:
                tr = {
                    "center": _bbox_center(bboxes[i]),
                    "cls": int(classes[i]),
                    "votes": deque([int(raw_statuses[i])], maxlen=self.config.smoothing_window),
                    "age": 1,
                    "gap": 0,
                }
                self._tracks.append(tr)
                updated_ids.add(len(self._tracks) - 1)

            if tr["age"] < self.config.min_track_age:
                final[i] = 0
            else:
                votes = tr["votes"]
                final[i] = 1 if sum(votes) * 2 > len(votes) else 0

        # Increment gaps of unmatched tracks and discard expired tracks.
        kept = []
        for t_idx, tr in enumerate(self._tracks):
            if t_idx in updated_ids:
                kept.append(tr)
            else:
                tr["gap"] += 1
                if tr["gap"] <= self.config.max_gap_frames:
                    kept.append(tr)
        self._tracks = kept

        return final

    def _push_state(self, gray_ds, bboxes, classes, ds):
        centers = _bbox_centers(bboxes) if len(bboxes) > 0 else None
        min_dist = max(5, int(30 // ds))
        features = cv2.goodFeaturesToTrack(gray_ds, maxCorners=100, qualityLevel=0.02, minDistance=min_dist)
        self._history.append((
            gray_ds.copy(),
            bboxes.copy(),
            classes.copy(),
            centers,
            features,
        ))

    def _is_edge_clipped(self, bbox: np.ndarray, img_h: int, img_w: int) -> bool:
        x1, y1, x2, y2 = bbox
        eps = self.config.edge_motion_guard_px
        return x1 < eps or y1 < eps or x2 > img_w - eps or y2 > img_h - eps


def _bbox_center(bbox: np.ndarray) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])


def _bbox_centers(bboxes: np.ndarray) -> np.ndarray:
    """Convert (N, 4) xyxy boxes to (N, 2) centers without Python loops."""
    if len(bboxes) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return (bboxes[:, :2] + bboxes[:, 2:4]) / 2.0


def _bbox_diagonal(bbox: np.ndarray) -> float:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return float(np.sqrt(w * w + h * h))


def _bbox_diagonals(bboxes: np.ndarray) -> np.ndarray:
    """Compute diagonal lengths of (N, 4) xyxy boxes."""
    if len(bboxes) == 0:
        return np.empty(0, dtype=np.float64)
    wh = bboxes[:, 2:4] - bboxes[:, :2]
    return np.sqrt((wh ** 2).sum(axis=1))


def _greedy_match(
    src_centers: np.ndarray,
    src_classes: np.ndarray,
    dst_centers: np.ndarray,
    dst_classes: np.ndarray,
    max_dists: np.ndarray,
) -> dict[int, int]:
    """Exclusive class-aware greedy matching. Visit same-class pairs within their matching radii in distance order, using each source and destination once. Returns a source-index to destination-index mapping. Complexity: O(S*D*log(S*D))."""
    if len(src_centers) == 0 or len(dst_centers) == 0:
        return {}

    diff = src_centers[:, None, :] - dst_centers[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))  # (S, D)

    valid = (src_classes[:, None] == dst_classes[None, :]) & (dists < max_dists[:, None])
    pairs = np.argwhere(valid)
    if len(pairs) == 0:
        return {}

    order = np.argsort(dists[pairs[:, 0], pairs[:, 1]])
    result: dict[int, int] = {}
    used_dst: set[int] = set()
    for k in order:
        s, d = int(pairs[k, 0]), int(pairs[k, 1])
        if s in result or d in used_dst:
            continue
        result[s] = d
        used_dst.add(d)
    return result


def _warp_point(pt: np.ndarray, M: np.ndarray) -> np.ndarray:
    return M[:2, :2] @ pt + M[:, 2]
