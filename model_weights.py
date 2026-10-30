"""One source of truth for the model repository and local weight filename."""
from pathlib import Path

HF_REPO_ID = "Has1tas/yolo26s-aerial-uav"
WEIGHTS_FILENAME = "yolo26s-aerial.pt"
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / WEIGHTS_FILENAME


def ensure_weights():
    """Use local weights when available; otherwise download from the public Hub."""
    if DEFAULT_MODEL.is_file():
        return DEFAULT_MODEL
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=WEIGHTS_FILENAME,
                               local_dir=DEFAULT_MODEL.parent))


if __name__ == "__main__":
    print(ensure_weights())
