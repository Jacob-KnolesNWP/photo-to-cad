"""Separate the part from the white sheet, markers, and printed scale bar."""

from __future__ import annotations

import cv2
import numpy as np

from partcapture.pipeline.errors import CaptureError
from partcapture.pipeline.mosaic import Mosaic, world_to_pixel


def _fill_quad(mask: np.ndarray, quad_px: np.ndarray, value: int = 255) -> None:
    quad = np.round(quad_px).astype(np.int32)
    cv2.fillConvexPoly(mask, quad, value)


def marker_mask(mosaic: Mosaic) -> np.ndarray:
    """Pixels that belong to a marker we actually saw, expanded a little."""
    mask = np.zeros(mosaic.image.shape[:2], dtype=np.uint8)
    seen: set[int] = set()
    for view in mosaic.views:
        for obs in view.observations:
            if obs.marker_id in seen:
                continue
            key = (obs.layout_key, obs.sheet_index)
            pose = mosaic.poses.get(key)
            if pose is None:
                continue
            seen.add(obs.marker_id)
            corners = pose.apply(obs.corners_local_mm)
            center = corners.mean(axis=0)
            # Grow ~1.4 mm so the black border and the interpolation fringe are excluded.
            grown = center + (corners - center) * ((obs.size_mm + 2.8) / obs.size_mm)
            _fill_quad(mask, world_to_pixel(grown, mosaic))
    return mask


def _region_is_print(patch: np.ndarray, valid: np.ndarray) -> bool:
    """True when a known footer looks like black ink on white paper, not a wood part."""
    if patch.size == 0 or int(valid.sum()) < 20:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1][valid]
    val = hsv[:, :, 2][valid]
    if sat.size == 0:
        return False
    dark = float(np.mean(val < 90))
    bright = float(np.mean(val > 190))
    return float(np.median(sat)) < 40 and dark > 0.015 and bright > 0.35


def print_mask(mosaic: Mosaic) -> np.ndarray:
    """Mask scale bars and footer text when they are still visible as printing."""
    mask = np.zeros(mosaic.image.shape[:2], dtype=np.uint8)
    for key, inst in mosaic.instances.items():
        pose = mosaic.poses[key]
        layout = inst.layout
        rects = [
            # Footer band (scale bar + instructions).
            np.array(
                [
                    [0.0, 0.0],
                    [layout.page_width_mm, 0.0],
                    [layout.page_width_mm, layout.footer_height_mm],
                    [0.0, layout.footer_height_mm],
                ]
            ),
            # Scale bar itself, slightly padded, in case the footer test is borderline.
            np.array(
                [
                    [layout.scale_bar_x_mm - 2.0, layout.scale_bar_y_mm - layout.scale_bar_thickness_mm],
                    [
                        layout.scale_bar_x_mm + layout.scale_bar_mm + 2.0,
                        layout.scale_bar_y_mm - layout.scale_bar_thickness_mm,
                    ],
                    [
                        layout.scale_bar_x_mm + layout.scale_bar_mm + 2.0,
                        layout.scale_bar_y_mm + layout.scale_bar_thickness_mm + 8.0,
                    ],
                    [layout.scale_bar_x_mm - 2.0, layout.scale_bar_y_mm + layout.scale_bar_thickness_mm + 8.0],
                ]
            ),
        ]
        for rect in rects:
            quad = world_to_pixel(pose.apply(rect), mosaic)
            x0 = int(np.floor(np.min(quad[:, 0])))
            x1 = int(np.ceil(np.max(quad[:, 0])))
            y0 = int(np.floor(np.min(quad[:, 1])))
            y1 = int(np.ceil(np.max(quad[:, 1])))
            h, w = mask.shape
            x0c, x1c = max(0, x0), min(w, x1)
            y0c, y1c = max(0, y0), min(h, y1)
            if x1c <= x0c or y1c <= y0c:
                continue
            patch = mosaic.image[y0c:y1c, x0c:x1c]
            valid = mosaic.valid[y0c:y1c, x0c:x1c]
            if _region_is_print(patch, valid):
                _fill_quad(mask, quad)
    return mask


def segment_part(mosaic: Mosaic, user_mask: np.ndarray | None = None) -> np.ndarray:
    """Return a uint8 mask, 255 where the part is."""
    if user_mask is not None:
        mask = user_mask
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape != mosaic.image.shape[:2]:
            mask = cv2.resize(mask, (mosaic.image.shape[1], mosaic.image.shape[0]), interpolation=cv2.INTER_NEAREST)
        binary = np.where(mask > 127, 255, 0).astype(np.uint8)
        binary[~mosaic.valid] = 0
        return _largest_component(binary)

    hsv = cv2.cvtColor(mosaic.image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # White paper is bright and nearly neutral. Wood, cardboard, paint, and
    # dark gaskets are not. Black marker ink is removed separately.
    paper = (val >= 188) & (sat <= 42)
    part = mosaic.valid & ~paper
    blocked = marker_mask(mosaic) | print_mask(mosaic)
    part[blocked > 0] = False
    binary = np.where(part, 255, 0).astype(np.uint8)
    binary = _drop_specks(binary, mosaic.px_per_mm)
    binary = _largest_component(binary)
    if int(np.count_nonzero(binary)) < mosaic.px_per_mm * mosaic.px_per_mm * 8:
        raise CaptureError(
            "NO_PART",
            "No part was found on the sheet. Light or white parts blend into the paper — "
            "use Touch up to paint the outline. Also check that the part isn't covering every marker.",
        )
    return binary


def _drop_specks(mask: np.ndarray, px_per_mm: float) -> np.ndarray:
    min_area = max(6, int((1.2 * px_per_mm) ** 2))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == label] = 255
    return keep


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return np.where(labels == best, 255, 0).astype(np.uint8)


def contours_from_mask(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Outer contour and hole contours, in pixel coordinates (float)."""
    binary = (mask > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        raise CaptureError(
            "NO_PART",
            "The part outline came out empty. Use Touch up to paint the shape and try again.",
        )
    hierarchy = hierarchy[0]
    # External contours have parent == -1. Pick the largest.
    external = [i for i, h in enumerate(hierarchy) if h[3] == -1]
    if not external:
        raise CaptureError("NO_PART", "The part outline came out empty. Use Touch up to paint the shape.")
    outer_i = max(external, key=lambda i: cv2.contourArea(contours[i]))
    outer = contours[outer_i].reshape(-1, 2).astype(np.float64)
    holes = []
    child = hierarchy[outer_i][2]
    while child != -1:
        area = cv2.contourArea(contours[child])
        if area >= 6:
            holes.append(contours[child].reshape(-1, 2).astype(np.float64))
        child = hierarchy[child][0]
    return outer, holes
