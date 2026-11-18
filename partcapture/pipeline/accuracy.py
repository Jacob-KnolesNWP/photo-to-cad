"""Estimated measurement error.

The estimate has two parts that the UI shows separately:

* Capture quality — did the markers agree, and were there enough of them?
* A tolerance band that grows with the object's diagonal, because residual
  lens error, paper not being perfectly flat, and print error all scale up
  on a bigger part.

These are conservative shop numbers, not a lab calibration certificate.
Critical dimensions still get checked with calipers.
"""

from __future__ import annotations

import math

# Fraction of the part diagonal added (in quadrature) for unmodeled 2D error:
# residual lens distortion, sheet flatness, and print/pixel limits.
SIZE_FRACTION_2D = 0.0012  # 0.12%
# Each extra sheet is tied in by a homography; allow 0.20 mm per link.
SHEET_LINK_MM = 0.20
# Overlapping photos don't land on exactly the same pixel.
PHOTO_BLEND_MM = 0.04
RASTER_FLOOR_MM = 0.05

# 3D photogrammetry on a phone, CPU multi-view stereo, markers for scale.
# About ±1.5 mm on a small object, plus 0.30% of the diagonal.
BASE_3D_MM = 1.5
SIZE_FRACTION_3D = 0.003


def estimate_2d(
    reproj_mm: float,
    diagonal_mm: float,
    n_sheets: int,
    n_photos: int,
) -> dict:
    size_term = SIZE_FRACTION_2D * max(0.0, diagonal_mm)
    link_term = SHEET_LINK_MM * max(0, n_sheets - 1)
    photo_term = PHOTO_BLEND_MM * max(0, n_photos - 1)
    sigma = math.sqrt(
        reproj_mm**2 + size_term**2 + link_term**2 + photo_term**2 + RASTER_FLOOR_MM**2
    )
    return {
        "estimated_error_mm": sigma,
        "reproj_mm": reproj_mm,
        "size_term_mm": size_term,
        "sheet_link_term_mm": link_term,
        "photo_term_mm": photo_term,
        "diagonal_mm": diagonal_mm,
        "size_fraction": SIZE_FRACTION_2D,
    }


def estimate_2d_after_caliper(base: dict) -> dict:
    """Calipers anchored the scale, so the size term shrinks but does not vanish."""
    reproj = float(base["reproj_mm"])
    size_term = 0.5 * float(base["size_term_mm"])
    link_term = float(base["sheet_link_term_mm"])
    photo_term = float(base["photo_term_mm"])
    sigma = math.sqrt(reproj**2 + size_term**2 + link_term**2 + photo_term**2 + RASTER_FLOOR_MM**2)
    updated = dict(base)
    updated["estimated_error_mm"] = sigma
    updated["size_term_mm"] = size_term
    updated["anchored_to_calipers"] = True
    return updated


def estimate_3d(diagonal_mm: float, marker_scale_cv: float = 0.0) -> dict:
    """marker_scale_cv is the relative spread of per-marker scale samples (0.01 = 1%)."""
    size_term = SIZE_FRACTION_3D * max(0.0, diagonal_mm)
    marker_term = abs(marker_scale_cv) * max(0.0, diagonal_mm)
    sigma = math.sqrt(BASE_3D_MM**2 + size_term**2 + marker_term**2)
    return {
        "estimated_error_mm": sigma,
        "base_mm": BASE_3D_MM,
        "size_term_mm": size_term,
        "marker_term_mm": marker_term,
        "diagonal_mm": diagonal_mm,
        "size_fraction": SIZE_FRACTION_3D,
        "note": (
            "3D from phone photos is about ±1.5 mm on a small object and worse as the "
            "object gets larger. Verify every critical dimension with calipers before cutting."
        ),
    }


def capture_quality(
    *,
    reproj_mm: float,
    n_markers: int,
    min_sharpness: float,
    inlier_ratio: float,
) -> str:
    if n_markers >= 8 and reproj_mm <= 0.25 and min_sharpness >= 120 and inlier_ratio >= 0.8:
        return "good"
    if n_markers >= 4 and reproj_mm <= 0.6 and inlier_ratio >= 0.6:
        return "fair"
    return "poor"
