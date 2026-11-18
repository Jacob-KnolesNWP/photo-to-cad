"""End-to-end flat-part measurement."""

from __future__ import annotations

import math

import cv2
import numpy as np

from partcapture.pipeline.accuracy import capture_quality, estimate_2d
from partcapture.pipeline.detect import SHARPNESS_MIN, detect_markers, sharpness_score
from partcapture.pipeline.errors import CaptureError
from partcapture.pipeline.geometry import (
    FittedOutline,
    CirclePrim,
    fit_outline,
    hole_records,
    measure_points,
    primitive_points,
)
from partcapture.pipeline.mosaic import Mosaic, View, build_mosaic, pixel_to_world
from partcapture.pipeline.reference import correction_factors, scale_points, updated_accuracy
from partcapture.pipeline.segment import contours_from_mask, segment_part


def measure_flat_images(
    images: list[np.ndarray],
    user_mask: np.ndarray | None = None,
    measured_length_mm: float | None = None,
    measured_width_mm: float | None = None,
) -> dict:
    if not images:
        raise CaptureError("NO_PHOTOS", "Add at least one photo of the part on the marker sheet.")
    views, warnings = _views_from_images(images)
    mosaic = build_mosaic(views)
    for view in mosaic.views:
        warnings.extend(view.warnings)
    return measure_mosaic(
        mosaic,
        user_mask=user_mask,
        measured_length_mm=measured_length_mm,
        measured_width_mm=measured_width_mm,
        extra_warnings=warnings,
    )


def measure_mosaic(
    mosaic: Mosaic,
    user_mask: np.ndarray | None = None,
    measured_length_mm: float | None = None,
    measured_width_mm: float | None = None,
    extra_warnings: list[str] | None = None,
    precomputed_mask: np.ndarray | None = None,
) -> dict:
    warnings = list(extra_warnings or [])
    mask = precomputed_mask if precomputed_mask is not None else segment_part(mosaic, user_mask=user_mask)
    fitted, measurements, outer_mm, hole_mm = _fit_mask(mask, mosaic)
    accuracy = _accuracy(mosaic, measurements)
    quality = _quality(mosaic)
    correction = None
    if measured_length_mm is not None or measured_width_mm is not None:
        fitted, measurements, correction = _apply_reference(
            fitted,
            outer_mm,
            hole_mm,
            measurements,
            accuracy,
            measured_length_mm,
            measured_width_mm,
        )
        accuracy = updated_accuracy(accuracy)
        if correction["anisotropic"]:
            warnings.append(correction["warning"])
        else:
            warnings.append(correction["warning"])

    preview = render_preview(mosaic, fitted, measurements)
    marker_ids = sorted({obs.marker_id for view in mosaic.views for obs in view.observations})
    sheets = [
        {"layout": key[0], "sheet_number": key[1] + 1, "theta_deg": math.degrees(pose.theta), "tx_mm": pose.tx, "ty_mm": pose.ty}
        for key, pose in sorted(mosaic.poses.items())
    ]
    # Drop duplicate warnings, preserve order.
    deduped = []
    for warning in warnings:
        if warning and warning not in deduped:
            deduped.append(warning)
    return {
        "mosaic": mosaic.image,
        "mask": mask,
        "valid": mosaic.valid,
        "preview": preview,
        "fitted": fitted,
        "geometry": fitted.to_json(),
        "measurements": measurements,
        "accuracy": accuracy,
        "quality": quality,
        "warnings": deduped,
        "correction": correction,
        "origin_mm": list(mosaic.origin_mm),
        "px_per_mm": mosaic.px_per_mm,
        "width_mm": mosaic.width_mm,
        "height_mm": mosaic.height_mm,
        "marker_ids": marker_ids,
        "marker_count": len(marker_ids),
        "sheet_count": len(mosaic.poses),
        "photo_count": sum(1 for view in mosaic.views if view.mapping is not None),
        "sheets": sheets,
        "reproj_mm": max(view.mapping.rms_mm for view in mosaic.views if view.mapping is not None),
        "reproj_px": max(view.mapping.rms_px for view in mosaic.views if view.mapping is not None),
    }


def _views_from_images(images: list[np.ndarray]) -> tuple[list[View], list[str]]:
    warnings: list[str] = []
    views: list[View] = []
    blurry: list[int] = []
    empty: list[int] = []
    for index, image in enumerate(images):
        if image is None or image.size == 0:
            raise CaptureError("BAD_PHOTO", f"Photo {index + 1} could not be read. Use a JPEG or PNG.")
        observations = detect_markers(image)
        score = sharpness_score(image)
        # A blurry frame can still lock onto a marker and then mis-measure the edge.
        if score < SHARPNESS_MIN:
            blurry.append(index + 1)
            continue
        if len(observations) == 0:
            empty.append(index + 1)
            continue
        if len(observations) < 2:
            warnings.append(
                f"Photo {index + 1} only showed {len(observations)} marker. "
                "It needs at least two, so it was skipped."
            )
            continue
        views.append(View(index=index, image=image, observations=observations, sharpness=score))
    if not views:
        if blurry and len(blurry) == len(images):
            raise CaptureError(
                "BLURRY",
                "This photo looks blurry, so the outline would not be reliable. "
                "Hold the phone steady, tap the sheet to focus, and retake it in brighter light."
                if len(images) == 1
                else "These photos look blurry, so the outline would not be reliable. "
                "Hold the phone steady, tap the sheet to focus, and retake them in brighter light.",
            )
        if empty or blurry:
            raise CaptureError(
                "NO_MARKERS",
                "No calibration markers were found. Use a TrueShape marker sheet printed at 100% scale, "
                "lay it flat, and keep the black squares fully in the frame. Avoid glare.",
            )
        raise CaptureError(
            "TOO_FEW_MARKERS",
            "Not enough markers were readable. Keep at least four markers fully in the frame, "
            "in more than one row.",
        )
    if blurry:
        joined = ", ".join(str(n) for n in blurry)
        warnings.append(f"Skipped blurry photo {joined}. The measurement uses the sharper shots.")
    if empty:
        joined = ", ".join(str(n) for n in empty)
        warnings.append(f"No markers in photo {joined}; that shot was skipped.")
    unique = {obs.marker_id for view in views for obs in view.observations}
    if len(unique) < 4:
        raise CaptureError(
            "TOO_FEW_MARKERS",
            f"Only {len(unique)} marker{' was' if len(unique) == 1 else 's were'} readable "
            "(need at least 4). Move back a little so a whole corner of the sheet is in frame, "
            "and make sure the markers aren't covered by the part.",
        )
    return views, warnings


def _fit_mask(mask: np.ndarray, mosaic: Mosaic):
    outer_px, hole_px = contours_from_mask(mask)
    outer_mm = pixel_to_world(outer_px, mosaic)
    hole_mm = [pixel_to_world(h, mosaic) for h in hole_px]
    # Ignore tiny holes left by marker fringes. 4 mm across is below a real fastener hole
    # we'd trust; smaller ones are almost always segmentation noise.
    kept_holes = []
    for hole in hole_mm:
        meas = measure_points(hole)
        if meas["length_mm"] >= 3.0 and meas["width_mm"] >= 3.0:
            kept_holes.append(hole)
    fitted = fit_outline(outer_mm, kept_holes)
    if not fitted.outline:
        raise CaptureError(
            "NO_PART",
            "The outline couldn't be turned into geometry. Use Touch up to paint the part and try again.",
        )
    measurements = _measurements_from_fit(fitted)
    return fitted, measurements, outer_mm, kept_holes


def _measurements_from_fit(fitted: FittedOutline) -> dict:
    samples = primitive_points(fitted.outline)
    meas = measure_points(samples if len(samples) >= 3 else fitted.outer_points_mm)
    meas["holes"] = hole_records(fitted)
    diagonal = math.hypot(meas["length_mm"], meas["width_mm"])
    meas["diagonal_mm"] = float(diagonal)
    return meas


def _accuracy(mosaic: Mosaic, measurements: dict) -> dict:
    reproj = max(view.mapping.rms_mm for view in mosaic.views if view.mapping is not None)
    n_photos = sum(1 for view in mosaic.views if view.mapping is not None)
    return estimate_2d(
        reproj_mm=reproj,
        diagonal_mm=float(measurements["diagonal_mm"]),
        n_sheets=len(mosaic.poses),
        n_photos=n_photos,
    )


def _quality(mosaic: Mosaic) -> str:
    reproj = max(view.mapping.rms_mm for view in mosaic.views if view.mapping is not None)
    inlier = min(view.mapping.inlier_ratio for view in mosaic.views if view.mapping is not None)
    n_markers = len({obs.marker_id for view in mosaic.views for obs in view.observations})
    sharp = min((view.sharpness for view in mosaic.views if view.mapping is not None), default=0.0)
    return capture_quality(
        reproj_mm=reproj,
        n_markers=n_markers,
        min_sharpness=sharp,
        inlier_ratio=inlier,
    )


def _apply_reference(fitted, outer_mm, hole_mm, measurements, accuracy, measured_length, measured_width):
    factors = correction_factors(
        measurements["length_mm"],
        measurements["width_mm"],
        measured_length,
        measured_width,
    )
    center = np.array(measurements["center_mm"], dtype=np.float64)
    angle = float(measurements["angle_deg"])
    sl = factors["scale_length"]
    sw = factors["scale_width"]
    outer_s = scale_points(outer_mm, center, angle, sl, sw)
    holes_s = [scale_points(h, center, angle, sl, sw) for h in hole_mm]
    refit = fit_outline(outer_s, holes_s)
    new_meas = _measurements_from_fit(refit)
    # Keep the pre-scale diagonal's accuracy inputs; the caller replaces the number.
    correction = {
        "applied": True,
        "anisotropic": factors["anisotropic"],
        "scale_length": sl,
        "scale_width": sw,
        "uniform_scale": factors["uniform"],
        "length_before_mm": measurements["length_mm"],
        "width_before_mm": measurements["width_mm"],
        "length_after_mm": new_meas["length_mm"],
        "width_after_mm": new_meas["width_mm"],
        "estimated_error_before_mm": accuracy["estimated_error_mm"],
        "warning": (
            "Length and width disagreed by more than 0.8%, so the outline was stretched to both "
            "caliper readings. Hole diameters may have changed — check them."
            if factors["anisotropic"]
            else "Scale was adjusted to your caliper reading. The marker fit itself did not change."
        ),
    }
    return refit, new_meas, correction


def render_preview(mosaic: Mosaic, fitted: FittedOutline, measurements: dict) -> np.ndarray:
    canvas = mosaic.image.copy()
    # Dim areas the photos didn't cover so the sheet reads clearly.
    if mosaic.valid is not None:
        dim = canvas.astype(np.float32)
        dim[~mosaic.valid] *= 0.35
        canvas = dim.astype(np.uint8)
    _draw_outline(canvas, mosaic, fitted)
    _draw_dimensions(canvas, mosaic, measurements)
    return canvas


def _world_to_px(mosaic: Mosaic, points: np.ndarray) -> np.ndarray:
    origin_x, origin_y = mosaic.origin_mm
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    u = (pts[:, 0] - origin_x) * mosaic.px_per_mm - 0.5
    v = (origin_y + mosaic.height_mm - pts[:, 1]) * mosaic.px_per_mm - 0.5
    return np.round(np.column_stack([u, v])).astype(np.int32)


def _draw_outline(canvas: np.ndarray, mosaic: Mosaic, fitted: FittedOutline) -> None:
    samples = primitive_points(fitted.outline, step_deg=3.0)
    if len(samples) >= 2:
        poly = _world_to_px(mosaic, samples).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], isClosed=True, color=(20, 92, 220), thickness=2, lineType=cv2.LINE_AA)
    for hole in fitted.holes:
        if isinstance(hole, CirclePrim):
            center = _world_to_px(mosaic, hole.center.reshape(1, 2))[0]
            radius = max(1, int(round(hole.radius * mosaic.px_per_mm)))
            cv2.circle(canvas, tuple(int(v) for v in center), radius, (180, 90, 20), 2, lineType=cv2.LINE_AA)
        elif isinstance(hole, list):
            pts = primitive_points(hole)
            if len(pts) >= 2:
                poly = _world_to_px(mosaic, pts).reshape(-1, 1, 2)
                cv2.polylines(canvas, [poly], isClosed=True, color=(180, 90, 20), thickness=2, lineType=cv2.LINE_AA)


def _draw_dimensions(canvas: np.ndarray, mosaic: Mosaic, measurements: dict) -> None:
    length = measurements.get("length_mm", 0.0)
    width = measurements.get("width_mm", 0.0)
    label = f"{length:.1f} x {width:.1f} mm"
    origin_x, origin_y = mosaic.origin_mm
    # Put the label near the part center.
    center = np.array(measurements.get("center_mm", [origin_x, origin_y]), dtype=np.float64)
    px = _world_to_px(mosaic, center.reshape(1, 2))[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, mosaic.image.shape[1] / 1400.0)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    x = int(np.clip(px[0] - tw / 2, 8, canvas.shape[1] - tw - 8))
    y = int(np.clip(px[1] - 12, th + 8, canvas.shape[0] - 8))
    cv2.rectangle(canvas, (x - 6, y - th - 6), (x + tw + 6, y + 6), (255, 250, 240), -1)
    cv2.putText(canvas, label, (x, y), font, scale, (25, 25, 25), thickness, cv2.LINE_AA)
    for hole in measurements.get("holes", []):
        if hole.get("shape") != "circle":
            continue
        hp = _world_to_px(mosaic, np.array([[hole["cx_mm"], hole["cy_mm"]]]))[0]
        text = f"D {hole['diameter_mm']:.1f}"
        cv2.putText(
            canvas,
            text,
            (int(hp[0]) + 6, int(hp[1]) - 6),
            font,
            scale * 0.7,
            (140, 60, 10),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
