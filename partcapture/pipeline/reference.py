"""Apply one or two caliper measurements as a scale correction."""

from __future__ import annotations

import math

import cv2
import numpy as np

from partcapture.pipeline.accuracy import estimate_2d_after_caliper
from partcapture.pipeline.errors import CaptureError


def correction_factors(
    length_mm: float,
    width_mm: float,
    measured_length_mm: float | None,
    measured_width_mm: float | None,
) -> dict:
    """Uniform scale from one or two reference dimensions.

    When both are given and they disagree by more than 0.8%, an anisotropic
    correction is used along the part's length and width, and the result says so.
    """
    if measured_length_mm is None and measured_width_mm is None:
        raise CaptureError(
            "NO_REFERENCE",
            "Enter a caliper or tape measurement for the overall length, the overall width, or both.",
        )
    for name, value in (("length", measured_length_mm), ("width", measured_width_mm)):
        if value is None:
            continue
        if value <= 0 or value > 20000:
            raise CaptureError(
                "BAD_REFERENCE",
                f"The measured {name} must be a positive number in millimetres (got {value}).",
            )
    if length_mm <= 1 or width_mm <= 1:
        raise CaptureError(
            "BAD_REFERENCE",
            "The measured outline is too small to correct. Check the segmentation first.",
        )
    ratios = []
    if measured_length_mm is not None:
        ratios.append(("length", measured_length_mm / length_mm))
    if measured_width_mm is not None:
        ratios.append(("width", measured_width_mm / width_mm))
    if len(ratios) == 1:
        return {
            "scale_length": ratios[0][1],
            "scale_width": ratios[0][1],
            "anisotropic": False,
            "uniform": ratios[0][1],
        }
    sl = measured_length_mm / length_mm
    sw = measured_width_mm / width_mm
    spread = abs(sl - sw) / max(1e-9, 0.5 * (sl + sw))
    if spread <= 0.008:
        uniform = 0.5 * (sl + sw)
        return {
            "scale_length": uniform,
            "scale_width": uniform,
            "anisotropic": False,
            "uniform": uniform,
        }
    return {
        "scale_length": sl,
        "scale_width": sw,
        "anisotropic": True,
        "uniform": 0.5 * (sl + sw),
    }


def scale_points(
    points: np.ndarray,
    center_mm: np.ndarray,
    length_angle_deg: float,
    scale_length: float,
    scale_width: float,
) -> np.ndarray:
    """Scale world-mm points about center along the part's length and width axes."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if abs(scale_length - 1.0) < 1e-9 and abs(scale_width - 1.0) < 1e-9:
        return pts.copy()
    theta = math.radians(length_angle_deg)
    axis_l = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    axis_w = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    linear = scale_length * np.outer(axis_l, axis_l) + scale_width * np.outer(axis_w, axis_w)
    center = np.asarray(center_mm, dtype=np.float64)
    return (pts - center) @ linear.T + center


def warp_mask_to_reference(
    mask: np.ndarray,
    center_px: tuple[float, float],
    angle_deg: float,
    length_mm: float,
    width_mm: float,
    scale_length: float,
    scale_width: float,
    px_per_mm: float,
) -> np.ndarray:
    """Scale the mask about its center along the min-area-rectangle axes.

    angle_deg is the orientation of the *longer* side, in world/pixel +x/+y
    image coordinates (y down). OpenCV minAreaRect's angle is not used here;
    the caller passes the angle of the length axis in pixel space.
    """
    if abs(scale_length - 1) < 1e-6 and abs(scale_width - 1) < 1e-6:
        return mask
    theta = math.radians(angle_deg)
    # Pixel axes: +x right, +y down. Length axis unit vector.
    axis_l = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    axis_w = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    # If the stored angle was computed in y-up world, the caller converts it.
    affine = scale_length * np.outer(axis_l, axis_l) + scale_width * np.outer(axis_w, axis_w)
    center = np.array(center_px, dtype=np.float64)
    # p' = A (p - c) + c
    linear = affine
    offset = center - linear @ center
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = linear
    matrix[:, 2] = offset

    h, w = mask.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64)
    # Map image corners through the inverse so we know the required canvas.
    # Forward maps source -> dest. We warp the source mask with cv2.warpAffine,
    # which applies dest = M @ source. Expand the canvas if the part grows.
    grown = max(scale_length, scale_width, 1.0)
    pad = int(math.ceil(max(h, w) * (grown - 1) * 0.5 + 8 + 2 * px_per_mm))
    if pad > 0:
        mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        matrix = matrix.copy()
        matrix[0, 2] += pad
        matrix[1, 2] += pad
    out = cv2.warpAffine(
        mask,
        matrix,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return out


def describe_correction(before: dict, factors: dict, after_accuracy: dict) -> dict:
    return {
        "applied": True,
        "anisotropic": factors["anisotropic"],
        "scale_length": factors["scale_length"],
        "scale_width": factors["scale_width"],
        "uniform_scale": factors["uniform"],
        "length_before_mm": before["length_mm"],
        "width_before_mm": before["width_mm"],
        "estimated_error_before_mm": before.get("estimated_error_mm"),
        "estimated_error_mm": after_accuracy["estimated_error_mm"],
        "warning": (
            "Length and width disagreed by more than 0.8%, so the outline was stretched "
            "to both caliper readings. Circles may no longer be perfectly round — "
            "re-check hole diameters."
            if factors["anisotropic"]
            else "Scale was adjusted uniformly to your caliper reading."
        ),
    }


def updated_accuracy(accuracy: dict) -> dict:
    return estimate_2d_after_caliper(accuracy)
