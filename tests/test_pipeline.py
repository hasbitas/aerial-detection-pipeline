"""Verify class remapping and the public CSV/inference contracts on CPU."""

import csv
import io
import queue
import threading

import cv2
import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results

import run_pipeline as pipeline
import model_weights
from status_resolver import StatusResolver
from status_resolver.config import CLASS_NAMES, NAME_TO_SPEC_ID


def test_checkpoint_order_maps_to_canonical_ids():
    assert pipeline.build_class_map({0: "Car", 1: "Human", 2: "UAI", 3: "UAP"}) == {
        0: 0, 1: 1, 2: 3, 3: 2,
    }


@pytest.mark.parametrize("name,expected", list(NAME_TO_SPEC_ID.items()))
def test_all_legacy_names_normalize(name, expected):
    assert pipeline.build_class_map({9: f" {name} "}) == {9: expected}


def test_unknown_classes_fail_instead_of_silently_assigning_ids():
    with pytest.raises(ValueError, match="Unsupported checkpoint class"):
        pipeline.build_class_map({0: "Unknown"})


def test_natural_order():
    assert sorted(["frame10.png", "frame2.png", "frame1.png"], key=pipeline.natural_sort_key) == [
        "frame1.png", "frame2.png", "frame10.png",
    ]


def test_remapped_boxes_receive_correct_statuses_and_csv_names():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    names = {0: "Car", 1: "Human", 2: "UAI", 3: "UAP"}
    # A person is contained in UAP; UAI remains clear.
    boxes = np.array([[20, 20, 150, 150, .9, 3], [40, 40, 60, 70, .9, 1],
                      [200, 20, 350, 150, .9, 2], [200, 200, 250, 250, .9, 0]])
    result = Results(frame, path="frame1.png", names=names, boxes=torch.from_numpy(boxes))
    detections = StatusResolver().update(frame, result, class_map=pipeline.build_class_map(names))
    assert [(d["cls"], d["motion_status"], d["landing_status"]) for d in detections] == [
        (2, -1, 0), (1, -1, -1), (3, -1, 1), (0, 0, -1),
    ]
    handle = io.StringIO()
    pipeline.write_detection_rows(csv.writer(handle), "frame1.png", detections, fps=12.0)
    rows = list(csv.reader(io.StringIO(handle.getvalue())))
    assert [row[2] for row in rows] == ["UAP", "Person", "UAI", "Vehicle"]
    assert all(len(row) == len(pipeline.CSV_COLUMNS) for row in rows)
    annotated = pipeline.annotate_frame(result, detections, 12.0)
    assert annotated.shape == frame.shape
    assert np.any(annotated != frame)


def test_empty_frame_row_preserves_csv_shape():
    handle = io.StringIO()
    pipeline.write_detection_rows(csv.writer(handle), "empty.png", [], fps=2)
    assert next(csv.reader(io.StringIO(handle.getvalue()))) == ["empty.png", *([""] * 8), "2.00"]


@pytest.mark.parametrize("cls,motion,landing,label", [
    (0, 1, -1, "Vehicle 0.84 | moving"),
    (0, 0, -1, "Vehicle 0.84 | stationary"),
    (1, -1, -1, "Person 0.84"),
    (2, -1, 1, "UAP 0.84 | landing OK"),
    (2, -1, 0, "UAP 0.84 | landing blocked"),
    (3, -1, 1, "UAI 0.84 | landing OK"),
    (3, -1, 0, "UAI 0.84 | landing blocked"),
    (0, -1, -1, "Vehicle 0.84"),
    (2, -1, -1, "UAP 0.84"),
])
def test_annotations_show_only_applicable_statuses(monkeypatch, cls, motion, landing, label):
    frame = np.zeros((100, 400, 3), dtype=np.uint8)
    result = Results(frame, path="", names=CLASS_NAMES, boxes=np.empty((0, 6)))
    texts = []
    real_put_text = cv2.putText

    def record(image, text, *args):
        texts.append(text)
        return real_put_text(image, text, *args)

    monkeypatch.setattr(pipeline.cv2, "putText", record)
    det = {"bbox": [380, 0, 399, 80], "cls": cls, "conf": .84,
           "motion_status": motion, "landing_status": landing}
    annotated = pipeline.annotate_frame(result, [det], 12)
    assert texts == [label, "FPS: 12.00"]
    assert np.any(annotated[0, :380])  # Filled label is shifted inside the right edge.
    assert not np.any(frame)  # Rendering does not mutate the input.


def test_prefetch_propagates_decode_exceptions(tmp_path, monkeypatch):
    def fail(_):
        raise RuntimeError("Decode failed")
    monkeypatch.setattr(pipeline.cv2, "imread", fail)
    output = queue.Queue()
    pipeline._frame_reader([tmp_path / "frame.png"], output, threading.Event())
    assert isinstance(output.get_nowait(), RuntimeError)
    assert output.get_nowait() is None


@pytest.mark.parametrize("device,cuda_available,expected", [
    ("auto", False, "cpu"), ("auto", True, "cuda"),
    ("cpu", True, "cpu"), ("cuda", False, "cuda"),
])
def test_folder_pipeline_shares_settings_and_writes_profile(
        tmp_path, monkeypatch, device, cuda_available, expected):
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: cuda_available)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    source = tmp_path / "frames"
    source.mkdir()
    for name in ["frame10.png", "frame2.png"]:
        assert cv2.imwrite(str(source / name), frame)
    calls = []

    class Detector:
        def __call__(self, image, **kwargs):
            calls.append(kwargs)
            return [Results(image, path="", names=CLASS_NAMES, boxes=np.empty((0, 6)))]

    monkeypatch.setattr(pipeline, "load_detector", lambda _: (Detector(), {i: i for i in range(4)}))
    output = tmp_path / "nested" / "detections.csv"
    seen = []
    pipeline.run_yolo_on_frames("unused.pt", source, output, display=False, profile=True,
                                device=device,
                                on_frame=lambda name, image: seen.append(name))
    assert seen == ["frame2.png", "frame10.png"]
    assert all(c["device"] == expected and c["imgsz"] == 1280
               and c["half"] == (expected == "cuda") for c in calls)
    with output.open() as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == pipeline.CSV_COLUMNS
    assert len(rows) == 3
    with output.with_name("detections_profile.csv").open() as handle:
        profile = list(csv.reader(handle))
    assert all(len(row) == len(profile[0]) for row in profile)


def test_unreadable_frame_warns_and_continues(tmp_path, monkeypatch, caplog):
    for name in ("frame1.png", "frame3.png"):
        assert cv2.imwrite(str(tmp_path / name), np.zeros((80, 80, 3), np.uint8))
    (tmp_path / "frame2.png").write_bytes(b"invalid image")

    class Detector:
        def __call__(self, image, **kwargs):
            return [Results(image, path="", names=CLASS_NAMES, boxes=np.empty((0, 6)))]

    monkeypatch.setattr(pipeline, "load_detector", lambda _: (Detector(), {}))
    output = tmp_path / "detections.csv"
    pipeline.run_yolo_on_frames("unused.pt", tmp_path, output, display=False, device="cpu")
    assert "Cannot decode frame: frame2.png; skipping" in caplog.text
    assert any(record.levelname == "WARNING" for record in caplog.records)
    for path in (output, output.with_name("detections_profile.csv")):
        with path.open() as handle:
            rows = list(csv.reader(handle))
        assert [row[0] for row in rows[1:]] == ["frame1.png", "frame3.png"]


def test_cli_defaults_to_auto_and_training_image_size(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--frames", "frames"])
    args = pipeline.parse_args()
    assert (args.device, args.imgsz) == ("auto", 1280)


def test_local_weights_need_no_network(tmp_path, monkeypatch):
    weights = tmp_path / "weights.pt"
    weights.touch()
    monkeypatch.setattr(model_weights, "DEFAULT_MODEL", weights)
    assert model_weights.ensure_weights() == weights


def test_missing_weights_download_from_shared_repository(tmp_path, monkeypatch):
    import huggingface_hub
    weights = tmp_path / "weights.pt"
    monkeypatch.setattr(model_weights, "DEFAULT_MODEL", weights)
    assert model_weights.HF_REPO_ID == "Has1tas/yolo26s-aerial-uav"
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(weights)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert model_weights.ensure_weights() == weights
    assert calls == [{"repo_id": model_weights.HF_REPO_ID, "filename": model_weights.WEIGHTS_FILENAME,
                      "local_dir": weights.parent}]
