"""Metadata publication strips local paths while retaining scientific metadata."""

import hashlib
from types import SimpleNamespace

import torch

from hf.inspect_checkpoint import inspect_checkpoint


def test_inspection_redacts_nested_local_paths(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "weights.pt"
    checkpoint_path.write_bytes(b"trusted test fixture")
    network = SimpleNamespace(
        model=[SimpleNamespace(nl=3)], stride=torch.tensor([8, 16, 32]),
        names={0: "Car"}, yaml={"yaml_file": r"C:\training\yolo26s.yaml"},
    )
    metrics = {"metrics/mAP50(B)": 0.89934}
    checkpoint = {
        "ema": network, "epoch": 42, "version": "8.4.84",
        "train_args": {"model": "/private/training/last.pt", "data": "/dataset/data.yaml",
                       "imgsz": 1280, "batch": 4,
                       "nested": [r"C:\private\weights\best.pt"]},
        "train_metrics": metrics, "train_results": {"epoch": [1, 2]},
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    info = inspect_checkpoint(checkpoint_path)
    assert info["train_args"] == {
        "model": "last.pt", "data": "data.yaml", "imgsz": 1280, "batch": 4,
        "nested": ["best.pt"],
    }
    assert info["model_yaml"]["yaml_file"] == "yolo26s.yaml"
    assert info["train_metrics"] == metrics
    assert info["train_results"] == {"epoch": [1, 2]}
    assert (info["epoch"], info["ultralytics_version"]) == (42, "8.4.84")
    assert info["sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
