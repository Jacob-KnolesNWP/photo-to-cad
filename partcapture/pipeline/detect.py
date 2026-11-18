"""ArUco detection and blur checks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from partcapture.pipeline.errors import CaptureError
from partcapture.pipeline.sheets import MarkerSpec, get_dictionary, lookup_marker_id

# Sharpness is the 95th percentile of Laplacian variance over tiles, measured
# after resizing the long side to this length so phone resolution doesn't
# change the threshold.
SHARPNESS_EDGE = 1000
SHARPNESS_TILE = 80
# Tuned against synthetic sheets: a sharp photo is well above this, a
# Gaussian-blurred one (sigma ≈ 3 on the photo) falls below it.
SHARPNESS_MIN = 80.0


@dataclass
class MarkerObservation:
    marker_id: int
    layout_key: str
    sheet_index: int
    local_index: int
    corners_px: np.ndarray  # 4x2 float64, TL TR BR BL
    corners_local_mm: np.ndarray  # 4x2
    center_local_mm: np.ndarray
    size_mm: float


def make_detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    # Large parts photographed from farther away: markers can be small in frame.
    params.minMarkerPerimeterRate = 0.006
    params.maxMarkerPerimeterRate = 4.0
    return cv2.aruco.ArucoDetector(get_dictionary(), params)


def sharpness_score(image: np.ndarray) -> float:
    """Higher is sharper. Robust to a mostly-white sheet."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    long_edge = max(h, w)
    if long_edge > SHARPNESS_EDGE:
        scale = SHARPNESS_EDGE / long_edge
        gray = cv2.resize(
            gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    tile = SHARPNESS_TILE
    scores: list[float] = []
    th, tw = lap.shape[:2]
    for y in range(0, th - tile + 1, tile):
        for x in range(0, tw - tile + 1, tile):
            scores.append(float(lap[y : y + tile, x : x + tile].var()))
    if not scores:
        return float(lap.var()) if lap.size else 0.0
    return float(np.percentile(scores, 95))


def detect_markers(image: np.ndarray) -> list[MarkerObservation]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = make_detector()
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return []
    found: list[MarkerObservation] = []
    seen: set[int] = set()
    for marker_corners, marker_id_arr in zip(corners, ids.ravel()):
        marker_id = int(marker_id_arr)
        if marker_id in seen:
            continue
        seen.add(marker_id)
        located = lookup_marker_id(marker_id)
        if located is None:
            continue
        layout_key, sheet_index, spec = located
        pts = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
        found.append(
            MarkerObservation(
                marker_id=marker_id,
                layout_key=layout_key,
                sheet_index=sheet_index,
                local_index=spec.local_index,
                corners_px=pts,
                corners_local_mm=spec.corners_local(),
                center_local_mm=np.array([spec.x_mm, spec.y_mm], dtype=np.float64),
                size_mm=spec.size_mm,
            )
        )
    return found


def assert_photo_usable(image: np.ndarray, observations: list[MarkerObservation], photo_number: int) -> float:
    """Raise a clear error when a photo cannot be measured. Returns sharpness."""
    score = sharpness_score(image)
    if len(observations) == 0 and score < SHARPNESS_MIN:
        raise CaptureError(
            "BLURRY",
            f"Photo {photo_number} looks blurry, so the outline would not be reliable. "
            "Hold the phone steady, tap the sheet to focus, and retake it in brighter light.",
        )
    if len(observations) == 0:
        raise CaptureError(
            "NO_MARKERS",
            f"No calibration markers were found in photo {photo_number}. "
            "Use a TrueShape marker sheet printed at 100% scale, lay it flat, and keep "
            "the black squares fully in the frame. Avoid glare.",
        )
    return score


def marker_spec_size(spec: MarkerSpec) -> float:
    return spec.size_mm
