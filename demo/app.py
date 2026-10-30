"""Gradio demo for consecutive UAV frames or a video; CPU inference by default."""

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

# Support both `python demo/app.py` and Space execution from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CACHE = ROOT / ".gradio"
os.environ.setdefault("GRADIO_TEMP_DIR", str(CACHE / "cache"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import gradio as gr
from model_weights import ensure_weights
from run_pipeline import (
    CSV_COLUMNS, IMAGE_EXTENSIONS, annotate_frame, load_detector,
    natural_sort_key, process_frame, write_detection_rows,
)
from status_resolver import StatusConfig, StatusResolver

MAX_VIDEO_FRAMES = 600
MAX_IMAGES = 100
SAMPLE_VIDEO = ROOT / "examples" / "sample_300_frames.mp4"


def analyze(video, images, device="cpu"):
    """Process exactly one input sequence. Each request gets fresh temporal state."""
    if bool(video) == bool(images):
        raise gr.Error("Provide either a video or a sequence of images.")
    if images and len(images) > MAX_IMAGES:
        raise gr.Error(f"Upload at most {MAX_IMAGES} images per request.")

    work = CACHE / "runs" / uuid.uuid4().hex
    work.mkdir(parents=True)
    cap = None
    writer = None
    try:
        model, class_map = load_detector(ensure_weights())
        resolver = StatusResolver(StatusConfig())
        csv_path = work / "detections.csv"
        gallery = []
        video_path = None
        frame_count = 0
        if video:
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise ValueError("Cannot open the uploaded video.")
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if not 0 < source_fps <= 240:
                source_fps = 25.0

            def frames():
                for index in range(MAX_VIDEO_FRAMES + 1):
                    ok, frame = cap.read()
                    if not ok:
                        return
                    if index == MAX_VIDEO_FRAMES:
                        raise ValueError(f"Video exceeds the {MAX_VIDEO_FRAMES}-frame demo limit. Use the CLI for longer sequences.")
                    yield f"frame{index:06d}.png", frame
        else:
            paths = sorted((Path(p) for p in images), key=lambda p: natural_sort_key(p.name))

            def frames():
                for path in paths:
                    if path.suffix.lower() not in IMAGE_EXTENSIONS:
                        raise ValueError(f"Unsupported image type: {path.suffix}")
                    yield path.name, cv2.imread(str(path))

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow(CSV_COLUMNS)
            for name, frame in frames():
                if frame is None:
                    raise ValueError(f"Cannot decode image: {name}")
                result, detections, fps, _ = process_frame(model, resolver, class_map, frame, device=device)
                annotated = annotate_frame(result, detections, fps)
                write_detection_rows(csv_writer, name, detections, fps=fps)
                if video:
                    if writer is None:
                        video_path = work / "annotated_raw.mp4"
                        height, width = annotated.shape[:2]
                        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 source_fps, (width, height))
                        if not writer.isOpened():
                            raise ValueError("MP4 video encoding is unavailable on this system.")
                    writer.write(annotated)
                else:
                    image_path = work / f"annotated{frame_count:06d}.png"
                    if not cv2.imwrite(str(image_path), annotated):
                        raise ValueError("Cannot write annotated image.")
                    gallery.append((str(image_path), name))
                frame_count += 1
        if frame_count == 0:
            raise ValueError("The input contains no readable frames.")
        if video_path:
            # Encode browser-compatible H.264 explicitly rather than relying on
            # an optional system ffmpeg installation or browser MPEG-4 support.
            import imageio_ffmpeg
            writer.release()
            writer = None
            compatible_path = work / "annotated.mp4"
            subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(video_path),
                 "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(compatible_path)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            video_path.unlink()
            video_path = compatible_path
        return str(video_path) if video_path else None, gallery, str(csv_path)
    except Exception as exc:
        # Release handles before deleting partially generated artifacts on Windows.
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        shutil.rmtree(work, ignore_errors=True)
        raise gr.Error(str(exc)) from exc
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


def build_interface(device="cpu"):
    """Build the UI without loading weights, downloading files, or starting a server."""
    with gr.Blocks(delete_cache=(3600, 3600), title="Aerial Detection Pipeline") as app:
        gr.Markdown("# Aerial Detection Pipeline\nYOLO detections with motion and landing status. "
                    "Upload consecutive frames in filename order, or one short video. "
                    "This is a research demo, not a flight-safety system.")
        with gr.Row():
            video = gr.Video(label=f"Video (up to {MAX_VIDEO_FRAMES} frames)")
            images = gr.File(label=f"Frame sequence (up to {MAX_IMAGES} images)",
                             file_count="multiple", file_types=["image"], type="filepath")
        run = gr.Button("Analyze", variant="primary")
        output_video = gr.Video(label="Annotated video (no audio)", format="mp4")
        output_gallery = gr.Gallery(label="Annotated frames", columns=2)
        output_csv = gr.File(label="Download detection CSV")

        def predict(video, images):
            return analyze(video, images, device=device)

        run.click(predict, inputs=[video, images], outputs=[output_video, output_gallery, output_csv],
                  concurrency_limit=1)
        gr.Examples(
            examples=[[str(SAMPLE_VIDEO), None]], inputs=[video, images],
            outputs=[output_video, output_gallery, output_csv], fn=predict,
            cache_examples=False, run_on_click=True,
            label="Click the sample video example to analyze it (CPU may take a while)",
        )
        gr.Markdown("Outputs use canonical class IDs. Motion: -1 N/A, 0 stationary, 1 moving. "
                    "Landing: -1 N/A, 0 unsuitable, 1 suitable. "
                    "Image names define sequence order; unrelated photos do not yield meaningful motion.")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--open-browser", action="store_true", help="Open the local demo in your browser")
    args = parser.parse_args()
    # Custom output files live for the server lifetime; Gradio expires served cache copies.
    try:
        build_interface(args.device).queue(max_size=8).launch(inbrowser=args.open_browser)
    finally:
        shutil.rmtree(CACHE / "runs", ignore_errors=True)
