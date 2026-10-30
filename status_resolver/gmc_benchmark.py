"""
BoT-SORT GMC sufficiency benchmark.

Purpose: Determine whether BoT-SORT's built-in GMC (sparseOptFlow)
produces good enough ego-motion compensation to make a custom AKAZE
pipeline unnecessary.

What it measures:
- Residual displacement of KNOWN STATIONARY vehicles after GMC compensation.
  If GMC is working well, stationary objects should have near-zero residual.
- Comparison: GMC residual vs raw (uncompensated) residual.
  If GMC doesn't improve much over raw, it's not working.

Requirements:
- Video clips with both stationary and moving vehicles.
- Ground truth: annotated motion_status per track (manual labeling).
  Format: JSON file mapping frame_idx → {track_id: 0 or 1}

Usage:
    python -m status_resolver.gmc_benchmark \\
        --video path/to/clip.mp4 \\
        --model path/to/yolo26m.pt \\
        --gt path/to/ground_truth.json \\
        --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def run_benchmark(
    video_path: str,
    model_path: str,
    gt_path: str | None = None,
    out_path: str = "gmc_benchmark_results.json",
    max_frames: int = 0,
) -> dict:
    """
    Run the GMC sufficiency benchmark.

    Returns dict with per-track residual statistics.
    """
    # Late imports — don't require ultralytics just to import the module
    import cv2

    try:
        from ultralytics import YOLO
        from ultralytics.trackers.utils.gmc import GMC
    except ImportError:
        print("ERROR: ultralytics is required. pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    # Load model and GMC
    model = YOLO(model_path)
    gmc = GMC(method="sparseOptFlow", downscale=2)

    # Load ground truth if available
    gt = {}
    if gt_path and Path(gt_path).exists():
        with open(gt_path) as f:
            gt = json.load(f)
        # Convert string keys to int
        gt = {int(k): {int(tid): v for tid, v in frame_gt.items()}
              for k, frame_gt in gt.items()}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Per-track data collection
    track_data: dict[int, list[dict]] = defaultdict(list)
    prev_centers: dict[int, np.ndarray] = {}
    frame_idx = 0

    print(f"Processing: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break

        # Get GMC warp matrix
        warp_matrix = gmc.apply(frame)

        # Detect + track
        results = model.track(frame, persist=True, conf=0.25, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0 or boxes.id is None:
            frame_idx += 1
            continue

        bboxes = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int)
        img_h, img_w = frame.shape[:2]

        for i in range(len(track_ids)):
            if classes[i] != 0:  # Only vehicles
                continue

            tid = int(track_ids[i])
            bbox = bboxes[i]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            center = np.array([cx, cy])
            diag = float(np.sqrt((bbox[2]-bbox[0])**2 + (bbox[3]-bbox[1])**2))

            # Check if edge-clipped
            eps = 10
            is_clipped = (bbox[0] < eps or bbox[1] < eps or
                          bbox[2] > img_w - eps or bbox[3] > img_h - eps)

            if tid in prev_centers and not is_clipped:
                prev = prev_centers[tid]

                # Raw displacement (no compensation)
                raw_disp = float(np.linalg.norm(center - prev))

                # GMC-compensated displacement
                compensated_prev = warp_matrix[:2, :2] @ prev + warp_matrix[:, 2]
                gmc_disp = float(np.linalg.norm(center - compensated_prev))

                # Normalized
                raw_norm = raw_disp / diag if diag > 0 else 0
                gmc_norm = gmc_disp / diag if diag > 0 else 0

                # Ground truth if available
                gt_motion = gt.get(frame_idx, {}).get(tid, None)

                track_data[tid].append({
                    "frame": frame_idx,
                    "raw_residual": raw_norm,
                    "gmc_residual": gmc_norm,
                    "raw_px": raw_disp,
                    "gmc_px": gmc_disp,
                    "bbox_diag": diag,
                    "gt_motion": gt_motion,
                })

            if not is_clipped:
                prev_centers[tid] = center.copy()

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}...")

    cap.release()
    print(f"  Done. {frame_idx} frames processed, {len(track_data)} tracks.")

    # --- Analysis ---
    analysis = _analyze(track_data)

    # Save raw data + analysis
    output = {
        "video": video_path,
        "total_frames": frame_idx,
        "total_tracks": len(track_data),
        "analysis": analysis,
        "per_track_summary": {
            str(tid): {
                "n_observations": len(obs),
                "mean_raw_residual": float(np.mean([o["raw_residual"] for o in obs])),
                "mean_gmc_residual": float(np.mean([o["gmc_residual"] for o in obs])),
                "std_gmc_residual": float(np.std([o["gmc_residual"] for o in obs])),
                "p95_gmc_residual": float(np.percentile([o["gmc_residual"] for o in obs], 95)),
            }
            for tid, obs in track_data.items()
            if len(obs) > 2  # Skip very short tracks
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Print summary
    _print_summary(analysis)

    return output


def _analyze(track_data: dict[int, list[dict]]) -> dict:
    """Compute aggregate statistics."""
    all_raw = []
    all_gmc = []
    gt_stationary_gmc = []
    gt_moving_gmc = []

    for tid, observations in track_data.items():
        for o in observations:
            all_raw.append(o["raw_residual"])
            all_gmc.append(o["gmc_residual"])
            if o["gt_motion"] == 0:
                gt_stationary_gmc.append(o["gmc_residual"])
            elif o["gt_motion"] == 1:
                gt_moving_gmc.append(o["gmc_residual"])

    result = {
        "overall_mean_raw": float(np.mean(all_raw)) if all_raw else None,
        "overall_mean_gmc": float(np.mean(all_gmc)) if all_gmc else None,
        "overall_std_gmc": float(np.std(all_gmc)) if all_gmc else None,
        "compensation_ratio": None,
    }

    if all_raw and all_gmc:
        # How much GMC reduces residual: lower is better
        result["compensation_ratio"] = float(np.mean(all_gmc) / np.mean(all_raw))

    if gt_stationary_gmc:
        result["gt_stationary"] = {
            "n": len(gt_stationary_gmc),
            "mean": float(np.mean(gt_stationary_gmc)),
            "std": float(np.std(gt_stationary_gmc)),
            "p95": float(np.percentile(gt_stationary_gmc, 95)),
            "max": float(np.max(gt_stationary_gmc)),
        }

    if gt_moving_gmc:
        result["gt_moving"] = {
            "n": len(gt_moving_gmc),
            "mean": float(np.mean(gt_moving_gmc)),
            "p5": float(np.percentile(gt_moving_gmc, 5)),
            "min": float(np.min(gt_moving_gmc)),
        }

    # Separability: can we find a threshold between stationary and moving?
    if gt_stationary_gmc and gt_moving_gmc:
        stat_p95 = np.percentile(gt_stationary_gmc, 95)
        mov_p5 = np.percentile(gt_moving_gmc, 5)
        result["separability"] = {
            "stationary_p95": float(stat_p95),
            "moving_p5": float(mov_p5),
            "gap": float(mov_p5 - stat_p95),
            "separable": bool(mov_p5 > stat_p95),
            "suggested_thresh": float((stat_p95 + mov_p5) / 2) if mov_p5 > stat_p95 else None,
        }

    return result


def _print_summary(analysis: dict) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 60)
    print("GMC BENCHMARK SUMMARY")
    print("=" * 60)

    if analysis.get("compensation_ratio") is not None:
        ratio = analysis["compensation_ratio"]
        print(f"\nCompensation ratio (GMC/raw): {ratio:.3f}")
        if ratio < 0.3:
            print("  → GMC is significantly reducing residual. Good sign.")
        elif ratio < 0.7:
            print("  → GMC is partially effective.")
        else:
            print("  → GMC is NOT reducing residual much. May need custom pipeline.")

    if "gt_stationary" in analysis:
        s = analysis["gt_stationary"]
        print(f"\nStationary vehicles (n={s['n']}):")
        print(f"  Mean residual: {s['mean']:.4f}")
        print(f"  Std:           {s['std']:.4f}")
        print(f"  P95:           {s['p95']:.4f}")

    if "gt_moving" in analysis:
        m = analysis["gt_moving"]
        print(f"\nMoving vehicles (n={m['n']}):")
        print(f"  Mean residual: {m['mean']:.4f}")
        print(f"  P5:            {m['p5']:.4f}")

    if "separability" in analysis:
        sep = analysis["separability"]
        print(f"\nSeparability:")
        print(f"  Stationary P95: {sep['stationary_p95']:.4f}")
        print(f"  Moving P5:      {sep['moving_p5']:.4f}")
        print(f"  Gap:            {sep['gap']:.4f}")
        if sep["separable"]:
            print(f"  → SEPARABLE. Suggested threshold: {sep['suggested_thresh']:.4f}")
        else:
            print("  → NOT separable with single threshold. GMC may be insufficient.")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BoT-SORT GMC sufficiency for motion status."
    )
    parser.add_argument("--video", required=True, help="Path to video clip")
    parser.add_argument("--model", required=True, help="Path to YOLO26 weights")
    parser.add_argument("--gt", default=None, help="Path to ground truth JSON")
    parser.add_argument("--out", default="gmc_benchmark_results.json",
                        help="Output JSON path")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames to process (0 = all)")
    args = parser.parse_args()

    run_benchmark(args.video, args.model, args.gt, args.out, args.max_frames)


if __name__ == "__main__":
    main()
