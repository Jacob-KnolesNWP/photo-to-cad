"""Thickness correction and backing-paper detection on synthetic photos."""

import numpy as np

from partcapture.pipeline.mosaic import Pose2D
from partcapture.pipeline.photo_measure import measure_photo
from partcapture.pipeline.sheets import make_instance
from partcapture.synthetic import rectangle, render_photo, render_plane, work_area


def _sheet_photo(parts, eye, image_size=(1000, 800), focal_scale=1.2, px_per_mm=6):
    inst = make_instance("letter", 0)
    plane = render_plane([(inst, Pose2D())], parts, px_per_mm=px_per_mm)
    photo = render_photo(plane, eye, eye * np.array([1, 1, 0]), image_size=image_size, focal_scale=focal_scale)
    focal_px = focal_scale * max(image_size)
    return photo, focal_px


def test_close_camera_magnifies_until_thickness_is_applied():
    """A prism's top edge, shot from 200 mm, looks big. Thickness puts the footprint back."""
    left, bottom, right, top = work_area("letter", 0)
    cx, cy = (left + right) / 2, (bottom + top) / 2
    true_l, true_w, thickness, height = 86.0, 40.0, 14.0, 200.0
    # Silhouette on the sheet is the footprint scaled by Z/(Z-T) about the nadir.
    grow = height / (height - thickness)
    apparent = rectangle(cx - true_l / 2 * grow, cy - true_w / 2 * grow, cx + true_l / 2 * grow, cy + true_w / 2 * grow)
    photo, focal = _sheet_photo(
        [{"outer": apparent, "color": (40, 70, 150)}],
        np.array([cx, cy, height]),
        image_size=(1200, 1000),
        focal_scale=0.7,
    )
    bare = measure_photo(photo, focal_px=focal, thickness_mm=None, profile="prism")
    assert bare.ok, bare.error
    assert bare.raw_length_mm > true_l + 2.0
    assert bare.camera_height_mm < 300
    assert any("300 mm" in warning or "magnifies" in warning for warning in bare.warnings)

    fixed = measure_photo(photo, focal_px=focal, thickness_mm=thickness, profile="prism")
    assert fixed.ok, fixed.error
    assert abs(fixed.length_mm - true_l) < 1.5, (fixed.length_mm, fixed.raw_length_mm, fixed.camera_height_mm)
    assert abs(fixed.width_mm - true_w) < 1.5, (fixed.width_mm, fixed.raw_width_mm)


def test_rounded_profile_uses_half_thickness():
    left, bottom, right, top = work_area("letter", 0)
    cx, cy = (left + right) / 2, (bottom + top) / 2
    true_l, true_w, thickness, height = 80.0, 36.0, 12.0, 450.0
    grow = height / (height - thickness / 2)
    apparent = rectangle(cx - true_l / 2 * grow, cy - true_w / 2 * grow, cx + true_l / 2 * grow, cy + true_w / 2 * grow)
    photo, focal = _sheet_photo(
        [{"outer": apparent, "color": (50, 80, 140)}],
        np.array([cx, cy, height]),
    )
    result = measure_photo(photo, focal_px=focal, thickness_mm=thickness, profile="rounded")
    assert result.ok, result.error
    assert abs(result.length_mm - true_l) < 1.5, result.length_mm
    assert abs(result.width_mm - true_w) < 1.5, result.width_mm
    assert result.camera_height_mm > 300


def test_light_part_on_coloured_backing():
    left, bottom, right, top = work_area("letter", 0)
    cx, cy = (left + right) / 2, (bottom + top) / 2
    backing = rectangle(left + 8, bottom + 8, right - 8, top - 8)
    part = rectangle(cx - 25, cy - 15, cx + 25, cy + 15)
    photo, focal = _sheet_photo(
        [
            {"outer": backing, "color": (160, 70, 30)},
            {"outer": part, "color": (230, 225, 220)},
        ],
        np.array([cx, cy - 15, 480.0]),
        image_size=(1100, 900),
    )
    result = measure_photo(photo, focal_px=focal, bg="auto")
    assert result.ok, result.error
    assert abs(result.length_mm - 50) < 3.0, (result.length_mm, result.width_mm, result.warnings)
    assert abs(result.width_mm - 30) < 3.0, (result.length_mm, result.width_mm)
    # Must not have reported the backing sheet (~110 x 76).
    assert result.length_mm < 80
