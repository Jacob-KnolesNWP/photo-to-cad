"""Accuracy of the flat pipeline on synthetic photos with known dimensions.

Tolerances are in millimetres on the DXF / reported size. Synthetic pinhole
photos have no print error and no paper curl, so these limits are tighter than
the shop estimate the UI shows (that estimate grows with part size on purpose).
"""

import numpy as np
import cv2
import ezdxf
import pytest

from partcapture.pipeline.accuracy import estimate_2d
from partcapture.pipeline.dxf_export import dxf_extents_mm, write_dxf
from partcapture.pipeline.errors import CaptureError
from partcapture.pipeline.geometry import CirclePrim, LinePrim
from partcapture.pipeline.mosaic import Pose2D
from partcapture.pipeline.sheets import make_instance
from partcapture.pipeline.twod import measure_flat_images
from partcapture.synthetic import circle, rectangle, render_photo, render_plane, sample_flat_plate, work_area


def _dxf_circle_diameters(path: str) -> list[float]:
    doc = ezdxf.readfile(path)
    diameters = []
    for entity in doc.modelspace():
        if entity.dxftype() == "CIRCLE":
            diameters.append(float(entity.dxf.radius) * 2.0)
    return diameters


def test_letter_plate_and_hole_are_submillimetre(tmp_path):
    images, truth = sample_flat_plate()
    result = measure_flat_images(images)
    length = result["measurements"]["length_mm"]
    width = result["measurements"]["width_mm"]
    assert abs(length - truth["length_mm"]) < 0.40, length
    assert abs(width - truth["width_mm"]) < 0.40, width
    holes = result["measurements"]["holes"]
    assert len(holes) == 1 and holes[0]["shape"] == "circle"
    assert abs(holes[0]["diameter_mm"] - truth["hole_diameter_mm"]) < 0.30

    path = tmp_path / "plate.dxf"
    write_dxf(result["fitted"], path, unit="mm")
    dxf_w, dxf_h = dxf_extents_mm(path, "mm")
    # Extents follow the axis-aligned box of the geometry, which for this
    # axis-aligned plate matches length and width.
    assert abs(max(dxf_w, dxf_h) - truth["length_mm"]) < 0.40
    assert abs(min(dxf_w, dxf_h) - truth["width_mm"]) < 0.40
    diameters = _dxf_circle_diameters(str(path))
    assert len(diameters) == 1
    assert abs(diameters[0] - truth["hole_diameter_mm"]) < 0.30

    # Clean geometry: a rectangle plus a hole, not a noisy polyline.
    assert sum(isinstance(p, LinePrim) for p in result["fitted"].outline) <= 8
    assert isinstance(result["fitted"].holes[0], CirclePrim)
    assert result["accuracy"]["estimated_error_mm"] < 1.0
    assert result["quality"] == "good"


def test_inches_dxf_preserves_real_size(tmp_path):
    images, truth = sample_flat_plate()
    result = measure_flat_images(images)
    path = tmp_path / "plate_in.dxf"
    write_dxf(result["fitted"], path, unit="in")
    doc = ezdxf.readfile(str(path))
    assert doc.units == 1  # inches, $INSUNITS
    width, height = dxf_extents_mm(path, "in")
    assert abs(max(width, height) - truth["length_mm"]) < 0.40
    assert abs(min(width, height) - truth["width_mm"]) < 0.40


def test_mild_lens_distortion_stays_within_half_millimetre():
    inst = make_instance("letter", 0)
    left, bottom, right, top = work_area("letter", 0)
    cx = (left + right) / 2
    cy = (bottom + top) / 2
    outer = rectangle(cx - 40, cy - 25, cx + 40, cy + 25)
    plane = render_plane(
        [(inst, Pose2D())],
        [{"outer": outer, "holes": [], "color": (40, 80, 150)}],
        px_per_mm=8,
    )
    photo = render_photo(
        plane,
        np.array([cx, cy - 20, 480.0]),
        np.array([cx, cy, 0.0]),
        image_size=(1000, 1200),
        focal_scale=1.1,
        distortion_k1=-0.18,
    )
    result = measure_flat_images([photo])
    assert abs(result["measurements"]["length_mm"] - 80) < 0.50
    assert abs(result["measurements"]["width_mm"] - 50) < 0.50


def test_two_sheets_and_two_photos_stitch_a_long_plate():
    sheet_a = make_instance("letter", 0)
    sheet_b = make_instance("letter", 1)
    gap = 20.0
    pose_b = Pose2D(0.0, sheet_a.page_width_mm + gap, 0.0)
    left, bottom, right, top = work_area("letter", 0)
    far_right = pose_b.tx + work_area("letter", 1)[2] - 15
    x0 = left + 10
    y0 = (bottom + top) / 2 - 50
    y1 = (bottom + top) / 2 + 50
    outer = rectangle(x0, y0, far_right, y1)
    truth_length = far_right - x0
    truth_width = y1 - y0
    plane = render_plane(
        [(sheet_a, Pose2D()), (sheet_b, pose_b)],
        [{"outer": outer, "holes": [circle((x0 + far_right) / 2, (y0 + y1) / 2, 8.0)], "color": (50, 90, 160)}],
        px_per_mm=6,
    )
    cy = (y0 + y1) / 2
    photo_left = render_photo(
        plane,
        np.array([x0 + 40, cy - 30, 700.0]),
        np.array([x0 + 80, cy, 0.0]),
        (1000, 800),
        1.1,
    )
    photo_right = render_photo(
        plane,
        np.array([far_right - 40, cy + 20, 700.0]),
        np.array([far_right - 80, cy, 0.0]),
        (1000, 800),
        1.1,
    )
    result = measure_flat_images([photo_left, photo_right])
    assert result["sheet_count"] == 2
    assert result["photo_count"] == 2
    length = result["measurements"]["length_mm"]
    width = result["measurements"]["width_mm"]
    assert abs(length - truth_length) < 1.0, (length, truth_length)
    assert abs(width - truth_width) < 1.0, (width, truth_width)
    assert result["measurements"]["holes"][0]["shape"] == "circle"
    assert abs(result["measurements"]["holes"][0]["diameter_mm"] - 16.0) < 0.6
    # Larger than a letter-size part, so the reported band is wider, and still honest.
    assert result["accuracy"]["estimated_error_mm"] > 0.3


def test_caliper_reference_corrects_scale():
    images, _truth = sample_flat_plate()
    corrected = measure_flat_images(images, measured_length_mm=102.0, measured_width_mm=61.2)
    assert abs(corrected["measurements"]["length_mm"] - 102.0) < 0.15
    assert abs(corrected["measurements"]["width_mm"] - 61.2) < 0.15
    assert corrected["correction"]["applied"] is True
    assert corrected["correction"]["anisotropic"] is False


def test_blurry_photo_is_rejected():
    images, _truth = sample_flat_plate()
    blurry = cv2.GaussianBlur(images[0], (0, 0), 3.5)
    with pytest.raises(CaptureError) as caught:
        measure_flat_images([blurry])
    assert caught.value.code == "BLURRY"


def test_sharp_photo_without_markers_is_rejected():
    image = np.full((900, 700, 3), 240, np.uint8)
    cv2.rectangle(image, (80, 120), (520, 480), (30, 70, 130), -1)
    rng = np.random.default_rng(1)
    image = cv2.add(image, rng.integers(0, 25, image.shape, dtype=np.uint8))
    with pytest.raises(CaptureError) as caught:
        measure_flat_images([image])
    assert caught.value.code == "NO_MARKERS"


def test_estimated_error_grows_with_part_size():
    small = estimate_2d(reproj_mm=0.1, diagonal_mm=150, n_sheets=1, n_photos=1)
    large = estimate_2d(reproj_mm=0.1, diagonal_mm=2000, n_sheets=4, n_photos=6)
    assert small["estimated_error_mm"] < 0.5
    assert large["estimated_error_mm"] > 2.0
    assert large["estimated_error_mm"] > small["estimated_error_mm"] * 3
