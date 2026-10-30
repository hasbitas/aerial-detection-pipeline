"""Demo checks require the optional Gradio dependencies."""

import csv

import cv2
import numpy as np
import pytest

pytest.importorskip("gradio", reason="Install demo/requirements.txt to run demo checks")
from demo import app
from ultralytics.engine.results import Results


def test_interface_builds_without_loading_weights(monkeypatch):
    def fail():
        raise AssertionError("Building the UI must not fetch weights")
    monkeypatch.setattr(app, "ensure_weights", fail)
    interface = app.build_interface()
    assert interface.config["components"]
    interface.close()


def test_sample_example_runs_on_click_with_cpu_default(monkeypatch):
    registered = []
    real_examples = app.gr.Examples

    def record_examples(**kwargs):
        registered.append(kwargs)
        return real_examples(**kwargs)

    monkeypatch.setattr(app.gr, "Examples", record_examples)
    monkeypatch.setattr(app, "ensure_weights", lambda: pytest.fail("UI must not fetch weights"))
    monkeypatch.setattr(app, "analyze", lambda video, images, device: (video, images, device))
    interface = app.build_interface()
    try:
        example, = registered
        assert app.SAMPLE_VIDEO.is_file()
        assert example["examples"] == [[str(app.SAMPLE_VIDEO), None]]
        assert example["run_on_click"] is True
        assert example["cache_examples"] is False
        assert example["fn"](str(app.SAMPLE_VIDEO), None) == (str(app.SAMPLE_VIDEO), None, "cpu")
        assert len(example["inputs"]) == 2
        assert len(example["outputs"]) == 3
    finally:
        interface.close()


def test_demo_image_sequence_and_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(app, "ensure_weights", lambda: "unused.pt")
    monkeypatch.setattr(app, "load_detector", lambda _: (None, {}))

    def predict(model, resolver, mapping, frame, **kwargs):
        return Results(frame, path="", names={}, boxes=np.empty((0, 6))), [], 1.0, {}

    monkeypatch.setattr(app, "process_frame", predict)
    images = []
    for name in ["frame10.png", "frame2.png"]:
        path = tmp_path / name
        assert cv2.imwrite(str(path), np.zeros((80, 80, 3), np.uint8))
        images.append(str(path))
    video, gallery, csv_path = app.analyze(None, images)
    assert video is None
    assert [caption for _, caption in gallery] == ["frame2.png", "frame10.png"]
    with open(csv_path) as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 3
    assert all(len(row) == 10 for row in rows)


def test_demo_video_is_encoded_and_csv_matches_frames(tmp_path, monkeypatch):
    pytest.importorskip("imageio_ffmpeg")
    monkeypatch.setattr(app, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(app, "ensure_weights", lambda: "unused.pt")
    monkeypatch.setattr(app, "load_detector", lambda _: (None, {}))

    def predict(model, resolver, mapping, frame, **kwargs):
        return Results(frame, path="", names={}, boxes=np.empty((0, 6))), [], 1.0, {}

    monkeypatch.setattr(app, "process_frame", predict)
    source = tmp_path / "input.avi"
    encoder = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 80))
    assert encoder.isOpened()
    for value in (0, 60, 120):
        encoder.write(np.full((80, 80, 3), value, np.uint8))
    encoder.release()
    video, gallery, csv_path = app.analyze(str(source), None)
    assert gallery == []
    capture = cv2.VideoCapture(video)
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 3
    capture.release()
    with open(csv_path) as handle:
        assert len(list(csv.reader(handle))) == 4
