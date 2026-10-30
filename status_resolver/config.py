"""Canonical output classes and configurable status thresholds."""
from dataclasses import dataclass

CLS_VEHICLE = 0
CLS_PERSON = 1
CLS_UAP = 2
CLS_UAI = 3
CLASS_NAMES = {0: "Vehicle", 1: "Person", 2: "UAP", 3: "UAI"}

# All legacy non-English spellings belong in this normalization table only.
NAME_TO_SPEC_ID = {
    "car": 0, "vehicle": 0, "tasit": 0,
    "taşıt": 0, "Taşıt": 0,
    "human": 1, "person": 1, "insan": 1,
    "İnsan": 1, "i̇nsan": 1,
    "uap": 2, "uap alani": 2, "uap alanı": 2, "UAP Alanı": 2,
    "uai": 3, "uai alani": 3, "uai alanı": 3,
    "UAİ": 3, "UAİ Alanı": 3, "uai̇ alanı": 3, "uai̇": 3,
}

OBSTACLE_CLASSES = frozenset({CLS_VEHICLE, CLS_PERSON})
LANDING_CLASSES = frozenset({CLS_UAP, CLS_UAI})
MOTION_CLASSES = frozenset({CLS_VEHICLE})

@dataclass
class StatusConfig:
    debug_human_motion: bool = True
    """Compute debug person motion; official person motion remains -1."""
    edge_margin_px: int = 15
    """Landing boxes closer than this to the border are considered clipped."""
    containment_thresh: float = 0.3
    """Block landing if intersection / min(pad area, obstacle area) exceeds this."""
    normalized_residual_thresh: float = 0.05
    """Vehicle displacement after compensation divided by box diagonal."""
    human_residual_thresh: float = 0.05
    """Normalized displacement threshold for optional person debugging."""
    min_motion_px: float = 3.0
    """Absolute displacement floor to suppress small-box jitter."""
    noise_gain: float = 2.0
    """Multiply median background residual noise to obtain an adaptive pixel floor."""
    match_radius_diag: float = 1.5
    """Matching radius as a multiple of box diagonal."""
    motion_frame_skip: int = 3
    """Compare against this many frames back to accumulate displacement."""
    downscale_factor: int = 4
    """Optical flow downscale factor when adaptive scaling is disabled."""
    adaptive_downscale: bool = True
    """Target roughly 270 pixels in height for background optical flow."""
    smoothing_window: int = 5
    """Number of observations used for majority voting."""
    min_track_age: int = 2
    """Younger tracks default to stationary."""
    max_gap_frames: int = 10
    """Discard a track after this many missing frames."""
    edge_motion_guard_px: int = 10
    """Suppress raw motion updates for boxes close to an image edge."""
