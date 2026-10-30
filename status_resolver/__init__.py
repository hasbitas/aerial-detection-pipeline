"""Add motion_status and landing_status to YOLO detections using classical computer vision and lightweight matching."""

from .config import StatusConfig
from .resolver import StatusResolver

__all__ = ["StatusResolver", "StatusConfig"]
