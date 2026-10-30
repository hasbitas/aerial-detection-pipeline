"""Encode consecutive frame_NNNNNN.webp images as a browser-playable sample.

Example: python examples/make_sample.py path/to/frames --start 1200 --count 300
Requires ffmpeg on PATH, --ffmpeg, or the demo's imageio-ffmpeg dependency.
"""

import argparse
from pathlib import Path
import shutil
import subprocess


def make_sample(source, start, count, output, ffmpeg=None, crf=28):
    source, output = Path(source), Path(output)
    if start < 0 or count <= 0:
        raise ValueError("start must be nonnegative and count must be positive")
    for index in range(start, start + count):
        frame = source / f"frame_{index:06d}.webp"
        if not frame.is_file():
            raise FileNotFoundError(frame)
    if ffmpeg is None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(ffmpeg), "-y", "-framerate", "30", "-start_number", str(start),
        "-i", str(source / "frame_%06d.webp"), "-frames:v", str(count),
        "-vf", "scale=1280:720:flags=lanczos", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--start", type=int, default=1200)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "sample_300_frames.mp4")
    parser.add_argument("--ffmpeg", help="Optional ffmpeg executable path")
    parser.add_argument("--crf", type=int, default=28)
    args = parser.parse_args()
    make_sample(args.source, args.start, args.count, args.output, args.ffmpeg, args.crf)
