"""Count CPU detections in windows of a naturally sorted image sequence.

Samples zero-based positions 0, stride, 2*stride, ...; windows count source
positions, not sampled images. Counts are box observations, not unique objects.
Only reads the source; writes one JSON report to --output.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from model_weights import DEFAULT_MODEL
from run_pipeline import IMAGE_EXTENSIONS, load_detector, natural_sort_key
from status_resolver.config import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    if min(args.stride, args.window, args.imgsz) <= 0 or not 0 <= args.conf <= 1:
        parser.error("stride/window/imgsz must be positive; confidence must be in [0, 1]")
    if args.output.resolve().is_relative_to(args.frames.resolve()):
        parser.error("output must be outside the source folder")
    frames = sorted((p for p in args.frames.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                    key=lambda p: natural_sort_key(p.name))
    if not frames:
        parser.error("no supported source images")
    model, mapping = load_detector(args.model)
    windows = []
    for start in range(0, len(frames), args.window):
        end = min(start + args.window, len(frames))
        counts, samples = Counter(), []
        for index in range(start, end):
            if index % args.stride:
                continue
            frame = cv2.imread(str(frames[index]))
            if frame is None:
                raise ValueError(f"Cannot decode {frames[index]}")
            result = model(frame, device="cpu", conf=args.conf, imgsz=args.imgsz,
                           half=False, verbose=False)[0]
            counts.update(mapping[int(cls)] for cls in result.boxes.cls.cpu().tolist())
            samples.append(frames[index].name)
        row = {"start_index": start, "end_index": end - 1,
               "first_frame": frames[start].name, "last_frame": frames[end - 1].name,
               "sampled_frames": samples,
               "detections": {name: counts[cls] for cls, name in CLASS_NAMES.items()}}
        windows.append(row)
        print(f"{start:04d}-{end - 1:04d}: {row['detections']}", flush=True)
    report = {"source": str(args.frames.resolve()), "source_frames": len(frames),
              "device": "cpu", "stride": args.stride, "window": args.window,
              "confidence": args.conf, "imgsz": args.imgsz, "model": str(args.model),
              "counts_meaning": "Detection boxes in sampled frames; not unique objects or extrapolated totals",
              "windows": windows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
