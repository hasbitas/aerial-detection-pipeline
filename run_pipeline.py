"""Run YOLO and temporal status resolution over naturally sorted image frames."""

import argparse
import csv
import logging
from contextlib import ExitStack
from pathlib import Path
import queue
import re
import threading
import time

import cv2
import torch
from ultralytics import YOLO

from model_weights import DEFAULT_MODEL
from status_resolver import StatusConfig, StatusResolver
from status_resolver.config import CLASS_NAMES, NAME_TO_SPEC_ID

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
CSV_COLUMNS = ["frame_id", "class_id", "class_name", "x1", "y1", "x2", "y2",
               "motion_status", "landing_status", "fps"]
MOTION_LABELS = {-1: "N/A", 0: "Stationary", 1: "Moving"}
LANDING_LABELS = {-1: "N/A", 0: "Unsuitable", 1: "Suitable"}
LOGGER = logging.getLogger(__name__)


def natural_sort_key(value):
    """Sort frame2 before frame10, independently of letter case."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))]


def build_class_map(model_names):
    """Map checkpoint labels to output IDs; fail on unsupported class names."""
    names = model_names.items() if isinstance(model_names, dict) else enumerate(model_names)
    mapping = {}
    for model_id, name in names:
        label = str(name).strip()
        canonical_id = NAME_TO_SPEC_ID.get(label, NAME_TO_SPEC_ID.get(label.lower()))
        if canonical_id is None:
            raise ValueError(f"Unsupported checkpoint class: {name!r}")
        mapping[int(model_id)] = canonical_id
    return mapping


def load_detector(model_path):
    """Normalize model display names only after constructing the class mapping."""
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Weights not found: {model_path}. Run python model_weights.py.")
    model = YOLO(str(model_path))
    class_map = build_class_map(model.names)
    model.model.names = {key: CLASS_NAMES[value] for key, value in class_map.items()}
    return model, class_map


def annotate_frame(results, detections, fps):
    """Draw canonical classes and official motion/landing codes on a frame."""
    annotated = results.orig_img.copy()
    colors = {0: (80, 210, 255), 1: (255, 180, 80), 2: (80, 220, 80), 3: (220, 100, 220)}
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        color = colors[det["cls"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = (f"{CLASS_NAMES[det['cls']]} {det['conf']:.2f} "
                 f"motion:{MOTION_LABELS[det['motion_status']]} "
                 f"landing:{LANDING_LABELS[det['landing_status']]}")
        cv2.putText(annotated, label, (max(0, x1), max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(annotated, f"FPS: {fps:.2f}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return annotated


def write_detection_rows(csv_writer, frame_name, detections, model_names=None, fps=0.0):
    """Write one row per detection, or a blank detection row for an empty frame."""
    if not detections:
        csv_writer.writerow([frame_name, *([""] * 8), f"{fps:.2f}"])
    for det in detections:
        csv_writer.writerow([
            frame_name, det["cls"], CLASS_NAMES[det["cls"]],
            *map(int, det["bbox"]), det["motion_status"], det["landing_status"], f"{fps:.2f}",
        ])


def process_frame(model, resolver, class_map, frame, device="cpu", conf=0.25, imgsz=1280):
    """Shared inference path for the CLI and demo; resolver state belongs to one sequence."""
    start = time.perf_counter()
    result = model(frame, verbose=False, device=device, conf=conf, imgsz=imgsz,
                   half=(str(device) == "cuda"))[0]
    inference_end = time.perf_counter()
    detections = resolver.update(frame, result, class_map=class_map)
    end = time.perf_counter()
    fps = 1.0 / max(end - start, 1e-9)
    timing = {"yolo_ms": (inference_end - start) * 1000,
              "resolver_ms": (end - inference_end) * 1000, **resolver.last_timing}
    return result, detections, fps, timing


def _frame_reader(frames, out_queue, stop_event):
    """Prefetch decoded frames; bounded puts allow cancellation without leaking a thread."""
    def put(item):
        while not stop_event.is_set():
            try:
                out_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    try:
        for path in frames:
            if stop_event.is_set():
                return
            frame = cv2.imread(str(path))
            if not put((path.name, frame)):
                return
    except Exception as exc:
        put(exc)
    finally:
        put(None)


def run_yolo_on_frames(model_path, frames_folder, csv_output_path=None,
                       display=True, profile=True, device="auto", conf=0.25,
                       imgsz=1280, on_frame=None):
    """Process a frame directory; return the CSV path and propagate failures to the caller."""
    folder = Path(frames_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Frame folder not found: {folder}")
    frames = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                    key=lambda p: natural_sort_key(p.name))
    if not frames:
        raise ValueError(f"No supported images in: {folder}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    csv_path = Path(csv_output_path or Path("outputs") / f"{Path(model_path).stem}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    model, class_map = load_detector(model_path)
    resolver = StatusResolver(StatusConfig())
    frame_queue = queue.Queue(maxsize=8)
    stop_event = threading.Event()
    reader = threading.Thread(target=_frame_reader, args=(frames, frame_queue, stop_event), daemon=True)
    processed = 0
    print(f"Processing {len(frames)} frames on {device}; output: {csv_path}")
    with ExitStack() as stack:
        writer = csv.writer(stack.enter_context(csv_path.open("w", newline="", encoding="utf-8")))
        writer.writerow(CSV_COLUMNS)
        profiler = None
        if profile:
            profile_path = csv_path.with_name(csv_path.stem + "_profile.csv")
            profiler = csv.writer(stack.enter_context(profile_path.open("w", newline="", encoding="utf-8")))
            profiler.writerow(["frame_id", "num_detections", "yolo_ms", "resolver_ms",
                               "extract_ms", "egomotion_ms", "landing_ms", "assemble_ms", "fps"])
        reader.start()
        try:
            while True:
                item = frame_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                name, frame = item
                if frame is None:
                    LOGGER.warning("Cannot decode frame: %s; skipping", name)
                    continue
                result, detections, fps, timing = process_frame(
                    model, resolver, class_map, frame, device, conf, imgsz)
                write_detection_rows(writer, name, detections, fps=fps)
                if profiler:
                    profiler.writerow([name, len(detections), *timing.values(), f"{fps:.2f}"])
                processed += 1
                if display or on_frame:
                    annotated = annotate_frame(result, detections, fps)
                    if on_frame:
                        on_frame(name, annotated)
                    if display:
                        cv2.imshow("Aerial detection", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                if processed == 1 or processed % 25 == 0:
                    print(f"Processed {processed}/{len(frames)} frames ({fps:.1f} FPS)")
        finally:
            stop_event.set()
            reader.join(timeout=2)
            if display:
                cv2.destroyAllWindows()
    print(f"Completed {processed} frames; CSV: {csv_path}")
    return str(csv_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Weights path (default: models/yolo26s-aerial.pt)")
    parser.add_argument("--frames", required=True, help="Folder containing consecutive image frames")
    parser.add_argument("--csv", help="Output CSV path (default: outputs/<model_name>.csv)")
    parser.add_argument("--no-display", action="store_true", help="Disable the OpenCV preview window")
    parser.add_argument("--no-profile", action="store_true", help="Disable the per-frame timing CSV")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Inference device (default: auto = CUDA if available, otherwise CPU)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size (default: 1280)")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or args.imgsz <= 0:
        parser.error("--conf must be between 0 and 1 and --imgsz must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    run_yolo_on_frames(args.model, args.frames, args.csv, not args.no_display,
                       not args.no_profile, args.device, args.conf, args.imgsz)
