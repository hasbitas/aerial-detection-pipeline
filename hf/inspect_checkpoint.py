"""Extract stored metadata from the trusted local checkpoint without running inference.

Run from the repository root: python hf/inspect_checkpoint.py
Only load checkpoints you trust: weights_only=False enables Python pickle loading.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from model_weights import DEFAULT_MODEL


def redact_paths(value):
    """Keep only basenames for local paths, recursively, before publishing metadata."""
    if isinstance(value, dict):
        return {key: redact_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_paths(item) for item in value]
    if isinstance(value, (str, Path)):
        text = str(value).replace("\\", "/")
        if text.startswith(("/", "~/")) or (len(text) > 2 and text[1:3] == ":/"):
            return text.rstrip("/").rsplit("/", 1)[-1]
    return value


def inspect_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    network = checkpoint.get("ema")
    stored_in = "ema"
    if network is None:
        network = checkpoint.get("model")
        stored_in = "model"
    if network is None:
        raise ValueError("Neither ema nor model contains a network.")
    head = network.model[-1]
    strides = network.stride.tolist()
    with Path(path).open("rb") as handle:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
    result = {
        "checkpoint_filename": Path(path).name,
        "sha256": digest,
        "size_bytes": Path(path).stat().st_size,
        "source": "torch.load(map_location='cpu', weights_only=False); metadata only, no new evaluation",
        "checkpoint_keys": list(checkpoint),
        "network_stored_in": stored_in,
        "epoch": checkpoint.get("epoch"),
        "epoch_note": "Stored zero-based epoch; train_results records one-based epoch numbers.",
        "date": checkpoint.get("date"),
        "ultralytics_version": checkpoint.get("version"),
        "class_names": network.names,
        "train_args": checkpoint.get("train_args"),
        "model_yaml": network.yaml,
        "detect_head": {"type": type(head).__name__, "num_scales": head.nl, "strides": strides},
        "architecture_note": "The stored Detect head has strides 8, 16, 32 and no stride-4 P2 output; the legacy run name does not describe the architecture.",
        "train_metrics": checkpoint.get("train_metrics"),
        "train_results": checkpoint.get("train_results"),
        "metrics_note": "Stored training/validation metrics and epoch history; not an independent test-set evaluation. No per-class metrics are stored in these fields.",
    }
    if not result["train_metrics"] and not result["train_results"]:
        result["metrics_note"] = "No training metrics or training results are stored in this checkpoint."
    return redact_paths(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("checkpoint_info.json"))
    args = parser.parse_args()
    info = inspect_checkpoint(args.model)
    args.output.write_text(json.dumps(info, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Saved metadata to {args.output}")
