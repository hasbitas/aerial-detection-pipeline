"""Run the bundled video on CPU and regenerate documentation visuals and local stats.

Run from the repository root: python examples/render_demo.py
Uses existing local weights only. Measurements are observations, not accuracy scores.
"""

from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import imageio_ffmpeg

from model_weights import DEFAULT_MODEL
from run_pipeline import CSV_COLUMNS, annotate_frame, load_detector, process_frame, write_detection_rows
from status_resolver import StatusConfig, StatusResolver
from status_resolver.config import CLASS_NAMES


def main():
    output = ROOT / "outputs" / "sample_demo"
    output.mkdir(parents=True, exist_ok=True)
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    model, mapping = load_detector(DEFAULT_MODEL)
    resolver = StatusResolver(StatusConfig())
    capture = cv2.VideoCapture(str(ROOT / "examples" / "sample_300_frames.mp4"))
    writer = None
    counts = Counter()
    motion = Counter()
    landing = Counter()
    count, best_count, best_index, seconds = 0, -1, None, 0.0
    best_score = (-1, -1, -1, -1)
    frame_counts = []
    start = time.perf_counter()
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open sample video")
        with (output / "detections.csv").open("w", newline="", encoding="utf-8") as handle:
            rows = csv.writer(handle)
            rows.writerow(CSV_COLUMNS)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result, detections, fps, timing = process_frame(
                    model, resolver, mapping, frame, device="cpu")
                seconds += (timing["yolo_ms"] + timing["resolver_ms"]) / 1000
                annotated = annotate_frame(result, detections, fps)
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(output / "annotated_raw.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
                    if not writer.isOpened():
                        raise ValueError("Cannot encode annotated video")
                writer.write(annotated)
                write_detection_rows(rows, f"frame{count:06d}.png", detections, fps=fps)
                counts.update(det["cls"] for det in detections)
                motion.update(det["motion_status"] for det in detections if det["cls"] == 0)
                landing.update(det["landing_status"] for det in detections if det["cls"] in (2, 3))
                classes = Counter(det["cls"] for det in detections)
                vehicle_motion = Counter(det["motion_status"] for det in detections
                                         if det["cls"] == 0)
                mixed_motion = int(vehicle_motion[0] > 0 and vehicle_motion[1] > 0)
                frame_counts.append((classes[0], mixed_motion, classes[1]))
                # Prefer several vehicles with people, then a mix of motion labels.
                score = (int(classes[0] > 1 and classes[1] > 0),
                         classes[0], classes[1], mixed_motion)
                if score > best_score:
                    best_score = score
                    best_count, best_index = len(detections), count
                    if not cv2.imwrite(str(docs / "screenshot.png"), annotated):
                        raise ValueError("Cannot write screenshot")
                count += 1
                if count % 25 == 0:
                    print(f"{count} frames; cumulative pipeline FPS: {count / seconds:.2f}", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    elapsed = time.perf_counter() - start
    if count != 300:
        raise ValueError(f"Expected 300 frames, decoded {count}")
    # Maximize vehicle boxes over all contiguous six-second windows. Break ties
    # with frames containing both motion states, then person boxes; earliest wins.
    gif_frames = 180
    gif_start = max(range(count - gif_frames + 1), key=lambda start: tuple(
        sum(frame_counts[index][metric] for index in range(start, start + gif_frames))
        for metric in range(3)))
    stats = {
        "frames": count, "device": "cpu", "imgsz": 1280, "confidence": 0.25,
        "detections_per_class": {name: counts[key] for key, name in CLASS_NAMES.items()},
        "vehicle_motion_codes": dict(motion), "landing_codes": dict(landing),
        "pipeline_seconds": seconds, "pipeline_fps": count / seconds,
        "decode_annotate_write_seconds": elapsed, "end_to_end_fps": count / elapsed,
        "screenshot_video_frame": best_index, "screenshot_detections": best_count,
        "gif_start_video_frame": gif_start, "gif_frames": gif_frames,
        "gif_vehicle_detections": sum(row[0] for row in frame_counts[gif_start:gif_start + gif_frames]),
        "gif_mixed_motion_frames": sum(row[1] for row in frame_counts[gif_start:gif_start + gif_frames]),
    }
    (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    # Six seconds at natural speed; a small palette and ordered dithering limit size.
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(output / "annotated_raw.mp4"),
        "-ss", str(gif_start / 30), "-t", "6", "-filter_complex",
        "fps=8,scale=640:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer:bayer_scale=5",
        "-loop", "0", str(docs / "demo.gif"),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
