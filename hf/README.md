---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - yolo
  - uav
  - aerial
  - teknofest
---

# YOLO26s aerial UAV detector

Find vehicles, people, and marked parking or landing areas in images taken from above.

[Model weights](https://huggingface.co/Has1tas/yolo26s-aerial-uav) | [Browser demo setup and source code](https://github.com/hasbitas/aerial-detection-pipeline)

## What does it do?

This model draws boxes around four types of objects in aerial images.
It was used for TEKNOFEST 2026 Aviation AI, task 1.
For example: drone photo -> boxes labelled Vehicle, Person, UAP, or UAI.
The companion project can also estimate vehicle movement and whether a marked landing area looks clear.
Those extra statuses come from image-processing rules, not directly from this model.

## Try the demo

No coding is needed for the companion project's browser interface.

1. Install Python 3.11+ from [python.org](https://www.python.org/downloads/). On Windows tick **Add Python to PATH**.
2. Open the [source repository](https://github.com/hasbitas/aerial-detection-pipeline), click **Code -> Download ZIP**, and extract it.
3. On Windows double-click `setup.bat`, then `start.bat`. On macOS/Linux open Terminal in the extracted folder and run:
   ```sh
   chmod +x setup.sh start.sh
   ./setup.sh
   ./start.sh
   ```
4. Open **http://127.0.0.1:7860**, supply one aerial photo, consecutive images, or a short video, and click **Analyze**.
   The first analysis downloads `yolo26s-aerial.pt` (about 20.5 MB) unless it is already in the project's `models/` folder.
   No sample footage is bundled. CPU processing works without a graphics card and can take a while.
5. View the labelled images/video and download the CSV, a table that spreadsheet apps can open.
   Videos are limited to 600 frames; image requests to 100 images. Output video has no audio.
   Download results before stopping the server with **Ctrl+C** or closing its terminal.

Setup and the first weight download need internet. Later runs can use the local weights offline.
If Python is missing, reinstall with PATH enabled. If setup fails, read its last error and check your internet connection.
If port 7860 is busy, stop the earlier demo. If CUDA (NVIDIA's graphics-card computing system) is unavailable, use CPU.
See the [project README](https://github.com/hasbitas/aerial-detection-pipeline#troubleshooting) for more fixes.

## Classes and status codes

**UAP** means the TEKNOFEST flying-car parking area.
**UAI** means the TEKNOFEST flying-ambulance landing area.
They refer to competition markings, not every possible landing site.

| Stored checkpoint ID | Stored name | Pipeline output ID | English output name |
|---|---|---|---|
| 0 | Car | 0 | Vehicle |
| 1 | Human | 1 | Person |
| 2 | UAI | 3 | UAI (flying-ambulance landing area) |
| 3 | UAP | 2 | UAP (flying-car parking area) |

The saved model and the companion project number the two landing categories differently. The table shows both sets of numbers.


The companion project's CSV uses these codes:

| Code | Vehicle motion | Landing area |
|---|---|---|
| `-1` | Not applicable | Not applicable |
| `0` | Stationary | Unsuitable under the image rules |
| `1` | Moving | Suitable under the image rules |

Only Vehicle receives motion codes 0/1. Only UAP/UAI receive landing codes 0/1.
Person receives -1 for both. Other non-applicable fields also receive -1.
A single photo cannot establish motion; new or unmatched objects start as stationary.
"Suitable" is an image-based estimate, not proof that a real landing is safe.

## How it works

YOLO26s is an object detector, a trained program that finds objects and draws boxes around them.
Its weights, the saved information learned during training, are stored in `yolo26s-aerial.pt`.
The companion pipeline (a series of processing steps) compares consecutive frames to estimate motion,
then checks visible edges and obstacle overlap for landing areas.
This repository contains the model card and stored metadata; use the linked source repository for the demo.

## For developers

### Direct model use

Raw Ultralytics predictions retain checkpoint IDs. The companion pipeline maps by name before resolving statuses, so its output IDs differ for the two landing classes.

The example below runs from the companion repository root after installing its requirements.
Replace `frame.jpg` with your own existing image; it is not a bundled file.

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from model_weights import HF_REPO_ID, WEIGHTS_FILENAME

path = hf_hub_download(repo_id=HF_REPO_ID, filename=WEIGHTS_FILENAME)
model = YOLO(path)
results = model.predict("frame.jpg", imgsz=1280, device="cpu", conf=0.25)
```

To work outside the companion repository, pass `repo_id="Has1tas/yolo26s-aerial-uav"` and `filename="yolo26s-aerial.pt"` directly to `hf_hub_download`. Inference uses `imgsz=1280` to match the training image size. The example returns raw checkpoint classes; run `run_pipeline.py` from the linked repository to get canonical IDs and status CSV output.

### Architecture and stored training information

The run name contains `p2`, but **the inspected checkpoint has three detection scales with strides 8, 16, and 32, and no stride-4 P2 output**.
Strides describe how widely spaced the detector's output positions are in image pixels; P2 would be an extra scale for finer details.
Its embedded YAML (a text configuration) identifies `yolo26s.yaml`, scale `s`.
The run name `s_p2_1280_A` is a legacy label; this model is published as YOLO26s.

The following are extracted only from [checkpoint_info.json](checkpoint_info.json), using `torch.load(..., map_location="cpu", weights_only=False)` on the supplied checkpoint. No training or independent evaluation was performed for this card.

| Stored field | Value |
|---|---|
| Run name | `s_p2_1280_A` |
| Requested epochs (`train_args.epochs`) | 120 |
| Saved epoch (zero-based) | 42 |
| Epoch history | 43 entries, one-based epochs 1-43 |
| Image size | 1280 |
| Batch size | 4 |
| Dataset configuration | `data.yaml` |
| Model argument | Resumed `last.pt` |
| Embedded model YAML | `yolo26s.yaml`, scale `s`, 4 classes |
| Detection strides | 8, 16, 32 |
| Ultralytics version saved in checkpoint | 8.4.84 |
| Checkpoint date | 2026-07-02T15:16:08.195867 |
| Network storage | `ema`; `model` is null |

Stored `train_metrics` (aggregate validation metrics):

| Metric | Stored value |
|---|---|
| Precision (B) | 0.88789 |
| Recall (B) | 0.85498 |
| mAP50 (B) | 0.89934 |
| mAP50-95 (B) | 0.71862 |
| Validation box loss | 1.21325 |
| Validation class loss | 0.64283 |
| Validation DFL loss | 0.00198 |

`train_results` contains the epoch history and is preserved in the JSON. Requested epochs are not completed epochs. The fields do not establish dataset size, split composition, or per-class performance. The dataset and full training recipe are not distributed here.


These numbers summarize the stored validation run, not a new test on unseen footage.
Precision measures how many reported detections were correct. Recall measures how many labelled objects were found.
mAP summarizes detection quality across categories; mAP50 uses a looser box-overlap threshold than mAP50-95.
`B` means bounding boxes. Loss values measure training/validation error; smaller is generally better within the same setup.
A high overall score does not guarantee good results for each class or a new camera.

## Glossary

- **UAV:** an unmanned aerial vehicle, such as a drone.
- **Frame:** one still image from a video.
- **Class:** an object category, such as Person.
- **Bounding box:** the rectangle around a detected object.
- **Checkpoint:** a saved model file that may include training history.
- **Inference:** running a trained model on new images.
- **Validation:** checking a model on a separate set of labelled images during development.
- **Epoch:** one pass through the training data.
- **Batch size:** the number of training images processed together.
- **CSV:** a plain-text table that spreadsheet apps can open.

## Limitations

- Intended for research, portfolio demonstrations, and evaluation on aerial images. It is not a certified flight-control, emergency-response, or landing-safety system.
- Stored validation metrics should not be interpreted as independent test results.
- Earlier experiments on a previous dataset version (validation: 225 images, 361 instances) showed weak Person detection: YOLO12s Person P=1.00, R=0.02, mAP50=0.04; YOLO11s Person R=0.04, mAP50=0.07. These are owner-reported historical results, not scores for this YOLO26s checkpoint. Evaluate Person detection on your own footage before use.
- UAP and UAI are competition-specific classes, so general-purpose aerial detection is not established.
- Small objects, domain shifts, unseen cameras, lighting changes, and thermal imagery require fresh evaluation.
- Downstream motion and landing heuristics can fail when detections or camera-motion estimates fail. A clear detected box does not prove a safe real landing area.

## License and acknowledgements

[AGPL-3.0](https://github.com/hasbitas/aerial-detection-pipeline/blob/main/LICENSE).
Built with [Ultralytics](https://github.com/ultralytics/ultralytics) for the [TEKNOFEST](https://www.teknofest.org/) competition context.
An independent portfolio project by [hasbitas](https://github.com/hasbitas); no official endorsement is implied.
