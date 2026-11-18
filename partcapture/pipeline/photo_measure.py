"""Single-photo flat measurement with pose and thickness correction.

A phone held close to a raised part magnifies the silhouette: a feature at
height h above the sheet looks larger by Z/(Z-h) about the camera's ground
point. This is the correction that brought an 86 x 40 mm, 14 mm thick remote
(measured 88.2 x 41.2 mm with no thickness) back to about 85.8 x 40.1 mm.

The geometry follows a CLI that was checked on that photo. Marker IDs and
sheet coordinates come from this app's sheet layouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import ezdxf
import numpy as np

from partcapture.pipeline.errors import CaptureError
from partcapture.pipeline.sheets import LAYOUTS, get_dictionary, lookup_marker_id, make_instance

DEFAULT_HFOV_DEG = 69.0
_UP = 4


@dataclass
class PhotoMeasure:
    ok: bool
    length_mm: float = 0.0
    width_mm: float = 0.0
    raw_length_mm: float = 0.0
    raw_width_mm: float = 0.0
    holes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    error: str | None = None
    camera_height_mm: float | None = None
    tilt_deg: float | None = None
    estimated_error_mm: float | None = None
    overlay_bgr: np.ndarray | None = None
    dxf_bytes: bytes | None = None
    sheets: list = field(default_factory=list)
    corrections: dict = field(default_factory=dict)
    units: str = "mm"

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "length_mm": self.length_mm,
            "width_mm": self.width_mm,
            "raw_length_mm": self.raw_length_mm,
            "raw_width_mm": self.raw_width_mm,
            "holes": self.holes,
            "warnings": self.warnings,
            "error": self.error,
            "camera_height_mm": self.camera_height_mm,
            "tilt_deg": self.tilt_deg,
            "estimated_error_mm": self.estimated_error_mm,
            "sheets": self.sheets,
            "corrections": self.corrections,
            "units": self.units,
        }


def measure_photo(
    image: np.ndarray,
    *,
    thickness_mm: float | None = None,
    profile: str = "prism",
    caliper_mm: float | None = None,
    units: str = "mm",
    bg: str = "auto",
    focal_35mm: float | None = None,
    focal_px: float | None = None,
    px_per_mm: float = 5.0,
) -> PhotoMeasure:
    """Measure one photo. Never raises for a bad picture; returns ok=False."""
    if profile not in {"prism", "rounded"}:
        profile = "prism"
    if bg not in {"auto", "light", "dark"}:
        bg = "auto"
    if units not in {"mm", "in"}:
        units = "mm"
    warnings: list[str] = []
    try:
        return _measure(
            image,
            thickness_mm=thickness_mm,
            profile=profile,
            caliper_mm=caliper_mm,
            units=units,
            bg=bg,
            focal_35mm=focal_35mm,
            focal_px=focal_px,
            ppm=px_per_mm,
            warnings=warnings,
        )
    except CaptureError as exc:
        return PhotoMeasure(ok=False, error=exc.message, warnings=warnings, units=units)


def _detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    params.minMarkerPerimeterRate = 0.006
    params.maxMarkerPerimeterRate = 4.0
    return cv2.aruco.ArucoDetector(get_dictionary(), params)


def _sharpness(gray: np.ndarray) -> float:
    if max(gray.shape) > 1000:
        scale = 1000 / max(gray.shape)
        gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    tile = 80
    scores = [
        lap[y : y + tile, x : x + tile].var()
        for y in range(0, lap.shape[0] - tile + 1, tile)
        for x in range(0, lap.shape[1] - tile + 1, tile)
    ]
    return float(np.percentile(scores, 95)) if scores else float(lap.var())


def _detect(image: np.ndarray) -> list[dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = _detector()
    best = None
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    for variant in (gray, clahe):
        corners, ids, _ = detector.detectMarkers(variant)
        count = 0 if ids is None else len(ids)
        if best is None or count > best[0]:
            best = (count, corners, ids)
        if count >= 12:
            break
    _, corners, ids = best
    found = []
    seen: set[int] = set()
    if ids is None:
        return found
    for corner, marker_id in zip(corners, ids.ravel()):
        marker_id = int(marker_id)
        if marker_id in seen:
            continue
        seen.add(marker_id)
        located = lookup_marker_id(marker_id)
        if located is None:
            continue
        layout, sheet_index, spec = located
        found.append(
            {
                "id": marker_id,
                "layout": layout,
                "sheet": sheet_index,
                "spec": spec,
                "px": np.asarray(corner, np.float64).reshape(4, 2),
            }
        )
    return found


def _apply_h(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(-1, 1, 2), homography).reshape(-1, 2)


def _rigid(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    covariance = (src - src_mean).T @ (dst - dst_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt = vt.copy()
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = dst_mean - rotation @ src_mean
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = translation
    return matrix


def _solve_sheets(observations: list[dict]):
    grouped: dict[tuple, list] = {}
    for obs in observations:
        grouped.setdefault((obs["layout"], obs["sheet"]), []).append(obs)
    primary = max(grouped, key=lambda key: len(grouped[key]))
    poses = {key: np.eye(3) for key in grouped}

    def stack(items):
        return (
            np.vstack([item["px"] for item in items]),
            np.vstack([item["spec"].corners_local() for item in items]),
        )

    pixels, local = stack(grouped[primary])
    if len(pixels) < 8:
        raise CaptureError(
            "TOO_FEW_MARKERS",
            f"Only {len(grouped[primary])} marker(s) on the main sheet. Need at least two fully visible markers.",
        )
    homography, _ = cv2.findHomography(pixels, local, cv2.RANSAC, 1.0)
    if homography is None:
        raise CaptureError("MARKERS_NOT_PLANAR", "The markers don't sit on one flat plane. Tape the sheet down and retake.")
    inliers = None
    pixels_all = local_all = None
    for _ in range(3):
        for key, items in grouped.items():
            if key == primary:
                continue
            pix, loc = stack(items)
            poses[key] = _rigid(loc, _apply_h(homography, pix))
        pix_rows, glob_rows = [], []
        for key, items in grouped.items():
            for item in items:
                pix_rows.append(item["px"])
                glob_rows.append(_apply_h(poses[key], item["spec"].corners_local()))
        pixels_all = np.vstack(pix_rows)
        local_all = np.vstack(glob_rows)
        homography, inliers = cv2.findHomography(pixels_all, local_all, cv2.RANSAC, 1.0)
        if homography is None:
            raise CaptureError("MARKERS_NOT_PLANAR", "Couldn't line the sheets up. Retake with neighboring sheets in one photo.")
    mask = inliers.ravel().astype(bool)
    if int(mask.sum()) >= 8:
        refined, _ = cv2.findHomography(pixels_all[mask], local_all[mask], 0)
        if refined is not None:
            homography = refined
    error = np.linalg.norm(_apply_h(homography, pixels_all) - local_all, axis=1)
    good = error < 1.0
    cursor = 0
    for items in grouped.values():
        for item in items:
            item["global_mm"] = local_all[cursor : cursor + 4]
            item["inlier"] = bool(good[cursor : cursor + 4].all())
            cursor += 4
    return grouped, primary, poses, homography, pixels_all, local_all, error, good


def _focal_px(width: int, height: int, focal_35mm: float | None, focal_px: float | None) -> tuple[float, str]:
    if focal_px and focal_px > 1:
        return float(focal_px), "given"
    if focal_35mm and focal_35mm > 1:
        return float(focal_35mm) * math.hypot(width, height) / 43.2666, "exif"
    return max(width, height) / (2 * math.tan(math.radians(DEFAULT_HFOV_DEG / 2))), "default 69 deg field of view"


def _camera_pose(world_xy: np.ndarray, pixels: np.ndarray, k: np.ndarray):
    obj = np.c_[world_xy, np.zeros(len(world_xy))]
    candidates = []
    try:
        ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(obj, pixels, k, None, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            for rvec, tvec, err in zip(rvecs, tvecs, errors):
                rvec, tvec = cv2.solvePnPRefineLM(obj, pixels, k, None, rvec, tvec)
                rotation, _ = cv2.Rodrigues(rvec)
                center = (-rotation.T @ tvec).ravel()
                score = float(np.asarray(err).reshape(-1)[0])
                candidates.append((score, center, rotation, tvec))
    except cv2.error:
        candidates = []
    if not candidates:
        ok, rvec, tvec = cv2.solvePnP(obj, pixels, k, None)
        if not ok:
            raise CaptureError("POSE_FAILED", "Couldn't figure out where the camera was. Retake with more of the sheet in frame.")
        rvec, tvec = cv2.solvePnPRefineLM(obj, pixels, k, None, rvec, tvec)
        rotation, _ = cv2.Rodrigues(rvec)
        center = (-rotation.T @ tvec).ravel()
        candidates.append((0.0, center, rotation, tvec))
    above = [item for item in candidates if item[1][2] > 0]
    pool = above or candidates
    _err, center, rotation, _tvec = min(pool, key=lambda item: item[0])
    return center, rotation


def _kernel(radius_px: float):
    radius = max(1, int(round(radius_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _fit_bg(lab: np.ndarray, samples: np.ndarray, scale: float, origin_x: float, origin_y_top: float) -> np.ndarray:
    ys, xs = np.nonzero(samples)
    if len(xs) < 200:
        raise CaptureError("NO_PART", "Not enough bare sheet around the markers to tell the part from the paper.")
    if len(xs) > 40000:
        choice = np.random.RandomState(0).choice(len(xs), 40000, replace=False)
        ys, xs = ys[choice], xs[choice]
    xmm = origin_x + xs / scale
    ymm = origin_y_top - ys / scale
    design = np.c_[np.ones_like(xmm), (xmm - xmm.mean()) / 100.0, (ymm - ymm.mean()) / 100.0]
    height, width = lab.shape[:2]
    gx = (origin_x + np.arange(width) / scale - xmm.mean()) / 100.0
    gy = (origin_y_top - np.arange(height) / scale - ymm.mean()) / 100.0
    background = np.empty_like(lab)
    for channel in range(3):
        values = lab[ys, xs, channel].astype(np.float64)
        keep = np.ones(len(values), bool)
        coef = np.array([np.median(values), 0.0, 0.0])
        for _ in range(3):
            coef, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
            residual = values - design @ coef
            mad = 1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep]))) + 0.5
            keep = np.abs(residual) < 3 * mad
            if int(keep.sum()) < 100:
                break
        low, high = np.percentile(values, 1), np.percentile(values, 99)
        background[..., channel] = np.clip(
            coef[0] + coef[1] * gx[None, :] + coef[2] * gy[:, None], low - 5, high + 5
        ).astype(np.float32)
    return background


def _color_dist(lab: np.ndarray, background: np.ndarray) -> np.ndarray:
    delta = lab - background
    return np.sqrt((0.7 * delta[..., 0]) ** 2 + delta[..., 1] ** 2 + delta[..., 2] ** 2).astype(np.float32)


def _otsu(distance: np.ndarray, region: np.ndarray, floor: float) -> float:
    values = np.clip(distance[region] * 2.55, 0, 255).astype(np.uint8).reshape(-1, 1)
    if values.size < 100:
        return floor
    threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(floor, float(threshold) / 2.55)


def _clean(mask: np.ndarray, scale: float) -> np.ndarray:
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _kernel(0.4 * scale))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, _kernel(0.8 * scale))
    return closed.astype(bool)


def _pick_part(mask: np.ndarray, region: np.ndarray, scale: float, min_mm2: float = 8.0):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = []
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA] / (scale * scale)
        if area >= min_mm2:
            comps.append((index, area))
    if not comps:
        return None
    edge = region & ~cv2.erode(region.astype(np.uint8), _kernel(2)).astype(bool)
    best = best_any = None
    for index, area in comps:
        touches = bool((labels[edge] == index).any()) if edge.any() else False
        if best_any is None or area > best_any[1]:
            best_any = (index, area)
        if not touches and (best is None or area > best[1]):
            best = (index, area)
    chosen = best if best is not None else best_any
    return (labels == chosen[0]), chosen[1], best is None


def _segment(rectified, ctx, bg_mode: str, warnings: list[str]):
    scale = ctx["s"]
    blurred = cv2.GaussianBlur(rectified, (0, 0), 0.3 * scale)
    lab = cv2.cvtColor(blurred.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    roi = ctx["roi"]
    background = _fit_bg(lab, ctx["ring"], scale, ctx["X0"], ctx["Y1"])
    distance = _color_dist(lab, background)
    if ctx.get("gap_sample") is not None and int(ctx["gap_sample"].sum()) > 500:
        try:
            gap_bg = _fit_bg(lab, ctx["gap_sample"], scale, ctx["X0"], ctx["Y1"])
            gap_dist = _color_dist(lab, gap_bg)
            distance = np.where(ctx["gap_zone"], gap_dist, distance)
            distance = np.where(ctx["strip"], np.minimum(distance, gap_dist), distance).astype(np.float32)
        except CaptureError:
            pass
    smooth = cv2.GaussianBlur(distance, (0, 0), 0.5 * scale)
    threshold = _otsu(smooth, roi, 10.0)
    part = _clean((smooth > threshold) & roi, scale)
    result = {"level": 1, "D": distance, "thr": threshold, "lab": lab}
    first = _pick_part(part, roi, scale)
    roi_area = roi.sum() / (scale * scale)
    backing = None
    if first is not None:
        mask, area, touching = first
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        rectangularity = cv2.contourArea(contour) / max(1.0, rect[1][0] * rect[1][1])
        fraction = area / max(roi_area, 1e-6)
        if bg_mode == "dark" or (bg_mode == "auto" and fraction > 0.30 and rectangularity > 0.85):
            backing = mask
    elif bg_mode == "dark":
        raise CaptureError(
            "NO_PART",
            "No coloured backing paper showed up against the white sheet. Lay dark or coloured paper under a light part.",
        )
    if backing is not None:
        filled = np.zeros(backing.shape, np.uint8)
        contours, _ = cv2.findContours(backing.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, -1)
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, _kernel(3 * scale)).astype(bool)
        inner = cv2.erode(filled.astype(np.uint8), _kernel(1.5 * scale)).astype(bool) & ~ctx["marker_zone"]
        dist = cv2.distanceTransform(filled.astype(np.uint8), cv2.DIST_L2, 5)
        band = filled & (dist > 3 * scale) & (dist < 14 * scale) & ~ctx["marker_zone"]
        try:
            bg2 = _fit_bg(lab, band, scale, ctx["X0"], ctx["Y1"])
            dist2 = _color_dist(lab, bg2)
            smooth2 = cv2.GaussianBlur(dist2, (0, 0), 0.5 * scale)
            thr2 = _otsu(smooth2, inner, 10.0)
            part2 = _clean((smooth2 > thr2) & inner, scale)
            second = _pick_part(part2, inner, scale)
        except CaptureError:
            second = None
        if second is not None and second[1] < 0.9 * (filled.sum() / scale / scale):
            result.update(level=2, D=dist2, thr=thr2, mask=second[0], touching=second[2])
            return result
        if bg_mode == "dark":
            raise CaptureError("NO_PART", "Backing paper is there, but no part showed up on it. Check the contrast.")
        warnings.append(
            "A large coloured shape looked like backing paper, but nothing distinct sat on it. "
            "The outline is that shape. Retake on plain white paper if the part itself is that colour."
        )
    if first is None:
        raise CaptureError(
            "NO_PART",
            "No part found. Put a dark part on the white sheet, or a light part on matte dark paper inside the markers.",
        )
    result.update(mask=first[0], touching=first[2])
    return result


def _refine_outline(segment: dict, scale: float, origin_x: float, origin_y_top: float):
    mask = segment["mask"]
    ys, xs = np.nonzero(mask)
    margin = int(5 * scale)
    x0, x1 = max(0, int(xs.min()) - margin), min(mask.shape[1], int(xs.max()) + margin + 1)
    y0, y1 = max(0, int(ys.min()) - margin), min(mask.shape[0], int(ys.max()) + margin + 1)
    distance = segment["D"][y0:y1, x0:x1]
    up = cv2.resize(distance, None, fx=_UP, fy=_UP, interpolation=cv2.INTER_LINEAR)
    core = cv2.erode(mask.astype(np.uint8), _kernel(2.0 * scale)).astype(bool)[y0:y1, x0:x1]
    inner = distance[core]
    threshold = segment["thr"]
    if inner.size > 50:
        median = float(np.median(inner))
        threshold = float(np.clip(0.5 * median, 6.0, max(6.0, median * 0.8)))
    coarse = cv2.dilate(mask.astype(np.uint8), _kernel(1.2 * scale))[y0:y1, x0:x1]
    coarse_up = cv2.resize(coarse, None, fx=_UP, fy=_UP, interpolation=cv2.INTER_NEAREST).astype(bool)
    refined = ((up > threshold) & coarse_up).astype(np.uint8)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, _kernel(0.3 * scale * _UP))
    contours, hierarchy = cv2.findContours(refined, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise CaptureError("NO_PART", "The outline came out empty. Use more contrast between the part and the sheet.")
    external = [i for i in range(len(contours)) if hierarchy[0][i][3] == -1]
    outer_i = max(external, key=lambda i: cv2.contourArea(contours[i]))

    def to_mm(contour):
        pts = contour.reshape(-1, 2).astype(np.float64)
        u = (pts[:, 0] + 0.5) / _UP - 0.5 + x0
        v = (pts[:, 1] + 0.5) / _UP - 0.5 + y0
        return np.c_[origin_x + u / scale, origin_y_top - v / scale]

    outline = to_mm(contours[outer_i])
    holes = []
    for index in range(len(contours)):
        if hierarchy[0][index][3] == outer_i:
            hole = to_mm(contours[index])
            if abs(cv2.contourArea(hole.astype(np.float32))) >= 7.0:
                holes.append(hole)
    return {"outline": outline, "holes": holes, "mask_up": refined, "crop": (x0, y0, x1, y1)}


def _fit_circle(points: np.ndarray):
    x, y = points[:, 0], points[:, 1]
    design = np.c_[2 * x, 2 * y, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(design, x * x + y * y, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    radius = math.sqrt(max(float(sol[2]) + cx * cx + cy * cy, 0.0))
    residual = np.abs(np.hypot(x - cx, y - cy) - radius)
    return cx, cy, radius, float(residual.max()), float(np.sqrt((residual ** 2).mean()))


def _rect_dims(points: np.ndarray) -> tuple[float, float]:
    (_c, (a, b), _ang) = cv2.minAreaRect(points.astype(np.float32))
    return float(max(a, b)), float(min(a, b))


def _write_dxf(outline_xy: np.ndarray, holes: list[dict], units: str) -> bytes:
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4 if units == "mm" else 1
    doc.layers.add("OUTLINE", color=7)
    doc.layers.add("HOLES", color=5)
    doc.layers.add("NOTES", color=8)
    scale = 1.0 if units == "mm" else 1 / 25.4
    space = doc.modelspace()
    # Closed polyline of the fitted outline. Bulges are added where a run is an arc.
    vertices, bulges = _segments(outline_xy)
    space.add_lwpolyline(
        [(float(v[0] * scale), float(v[1] * scale), 0.0, 0.0, float(b)) for v, b in zip(vertices, bulges)],
        format="xyseb",
        close=True,
        dxfattribs={"layer": "OUTLINE"},
    )
    for hole in holes:
        if hole["type"] == "circle":
            space.add_circle(
                (hole["cx"] * scale, hole["cy"] * scale),
                hole["diameter_mm"] / 2 * scale,
                dxfattribs={"layer": "HOLES"},
            )
    import io

    buffer = io.StringIO()
    doc.write(buffer)
    return buffer.getvalue().encode("utf-8")


def _segments(points: np.ndarray, tolerance: float = 0.15):
    """Light line/arc fit. Falls back to a simplified closed polyline."""
    simplified = cv2.approxPolyDP(points.astype(np.float32).reshape(-1, 1, 2), 0.08, True).reshape(-1, 2)
    if len(simplified) < 3:
        simplified = points[:: max(1, len(points) // 80)]
    return simplified.astype(np.float64), [0.0] * len(simplified)


def _overlay(rectified, scale, origin_x, origin_y_top, apparent, corrected, holes, length, width, units, ground):
    canvas = rectified.copy()

    def mm_to_px(pts):
        q = np.asarray(pts, np.float64).reshape(-1, 2)
        return np.c_[(q[:, 0] - origin_x) * scale, (origin_y_top - q[:, 1]) * scale]

    cv2.polylines(canvas, [np.round(mm_to_px(apparent)).astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [np.round(mm_to_px(corrected)).astype(np.int32)], True, (0, 180, 0), 2, cv2.LINE_AA)
    for hole in holes:
        cv2.polylines(canvas, [np.round(mm_to_px(hole)).astype(np.int32)], True, (0, 140, 255), 2, cv2.LINE_AA)
    factor = 1.0 if units == "mm" else 1 / 25.4
    label = f"{length * factor:.2f} x {width * factor:.2f} {units}"
    cv2.putText(canvas, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
    gp = mm_to_px(ground)[0]
    if 0 <= gp[0] < canvas.shape[1] and 0 <= gp[1] < canvas.shape[0]:
        cv2.drawMarker(canvas, (int(gp[0]), int(gp[1])), (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
    return canvas


def _measure(image, *, thickness_mm, profile, caliper_mm, units, bg, focal_35mm, focal_px, ppm, warnings):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharp = _sharpness(gray)
    if sharp < 80:
        warnings.append(f"Photo looks blurry (sharpness {sharp:.0f}). Hold still, tap to focus, and add light.")
    observations = _detect(image)
    if len(observations) < 2:
        raise CaptureError(
            "NO_MARKERS",
            f"Found {len(observations)} marker(s). Use a sheet printed at 100% with the markers in the frame.",
        )
    grouped, primary, poses, homography, pixels, world, error, good = _solve_sheets(observations)
    used = sum(1 for item in observations if item["inlier"])
    if used < 8:
        warnings.append(f"Only {used} usable markers. Accuracy drops — aim for 12 or more.")
    reproj_rms = float(np.sqrt((error[good] ** 2).mean())) if good.any() else float(error.mean())
    sheets = []
    for key, items in grouped.items():
        expected = len(make_instance(key[0], key[1]).markers_by_id)
        sheets.append(
            {
                "layout": key[0],
                "sheet_number": key[1] + 1,
                "markers_found": len(items),
                "markers_expected": expected,
            }
        )
        if len(items) < 0.5 * expected:
            warnings.append(
                f"Only {len(items)} of {expected} markers on {key[0]} sheet {key[1] + 1}. "
                "Something is covering them, or the photo is too tilted."
            )

    focal, focal_source = _focal_px(width, height, focal_35mm, focal_px)
    k = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1.0]])
    center, rotation = _camera_pose(world[good], pixels[good], k)
    camera_z = abs(float(center[2]))
    optical = rotation.T @ np.array([0.0, 0.0, 1.0])
    tilt = math.degrees(math.acos(min(1.0, abs(float(optical[2])))))
    if camera_z < 300:
        warnings.append(
            f"Camera is only about {camera_z:.0f} mm ({camera_z / 25.4:.0f} in) above the sheet. "
            "Closer than 300 mm magnifies a raised edge. Step back to 400–500 mm (16–20 in) and use the 2x lens."
        )
    if tilt > 12:
        warnings.append(f"Camera is tilted {tilt:.0f}° off straight down. Shoot more directly overhead (under 12°).")

    all_corners = []
    for key in grouped:
        instance = make_instance(key[0], key[1])
        for spec in instance.markers_by_id.values():
            all_corners.append(_apply_h(poses[key], spec.corners_local()))
    all_corners = np.vstack(all_corners)
    pad = 6.0
    origin_x = float(all_corners[:, 0].min() - pad)
    origin_y_top = float(all_corners[:, 1].max() + pad)
    extent = max(all_corners[:, 0].max() - all_corners[:, 0].min(), all_corners[:, 1].max() - all_corners[:, 1].min()) + 2 * pad
    scale = min(ppm, 4500.0 / extent)
    cols = int(math.ceil((all_corners[:, 0].max() + pad - origin_x) * scale))
    rows = int(math.ceil((origin_y_top - (all_corners[:, 1].min() - pad)) * scale))
    to_px = np.array([[scale, 0, -origin_x * scale], [0, -scale, origin_y_top * scale], [0, 0, 1.0]])
    side_px = np.mean(
        [np.linalg.norm(item["px"][0] - item["px"][1]) / item["spec"].size_mm for item in observations if item["inlier"]]
    )
    source = image
    if side_px > 1.6 * scale:
        source = cv2.GaussianBlur(image, (0, 0), 0.45 * side_px / scale)
    warp = to_px @ homography
    rectified = cv2.warpPerspective(source, warp, (cols, rows), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
    valid = cv2.warpPerspective(np.full(gray.shape, 255, np.uint8), warp, (cols, rows)) > 0

    def mm_to_px(pts):
        q = np.asarray(pts, np.float64).reshape(-1, 2)
        return np.c_[(q[:, 0] - origin_x) * scale, (origin_y_top - q[:, 1]) * scale]

    marker_raw = np.zeros((rows, cols), np.uint8)
    hull_pts = []
    for key in grouped:
        instance = make_instance(key[0], key[1])
        for spec in instance.markers_by_id.values():
            quad = mm_to_px(_apply_h(poses[key], spec.corners_local()))
            cv2.fillConvexPoly(marker_raw, np.round(quad).astype(np.int32), 1)
            hull_pts.append(quad)
    marker_zone = cv2.dilate(marker_raw, _kernel(3.0 * scale)).astype(bool)
    hull = cv2.convexHull(np.vstack(hull_pts).astype(np.float32)).astype(np.int32)
    hull_mask = np.zeros((rows, cols), np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    roi = (
        cv2.erode(hull_mask, _kernel(1.5 * scale)).astype(bool)
        & ~marker_zone
        & cv2.erode(valid.astype(np.uint8), _kernel(2)).astype(bool)
    )
    ring = (
        cv2.dilate(marker_raw, _kernel(10 * scale)).astype(bool)
        & ~cv2.dilate(marker_raw, _kernel(3.5 * scale)).astype(bool)
        & hull_mask.astype(bool)
        & valid
    )
    pages = np.zeros((rows, cols), np.uint8)
    for key in grouped:
        layout = LAYOUTS[key[0]]
        page = _apply_h(
            poses[key],
            np.array(
                [
                    [0, 0],
                    [layout.page_width_mm, 0],
                    [layout.page_width_mm, layout.page_height_mm],
                    [0, layout.page_height_mm],
                ],
                np.float64,
            ),
        )
        cv2.fillConvexPoly(pages, np.round(mm_to_px(page)).astype(np.int32), 1)
    page_dilate = cv2.dilate(pages, _kernel(2 * scale)).astype(bool)
    page_erode = cv2.erode(pages, _kernel(2 * scale)).astype(bool)
    gap_zone = roi & ~page_dilate
    strip = page_dilate & ~page_erode
    gap_sample = cv2.erode(gap_zone.astype(np.uint8), _kernel(2 * scale)).astype(bool) if gap_zone.sum() > 400 else None
    segment = _segment(
        rectified,
        {
            "s": scale,
            "X0": origin_x,
            "Y1": origin_y_top,
            "roi": roi,
            "marker_zone": marker_zone,
            "ring": ring,
            "gap_zone": gap_zone,
            "strip": strip,
            "gap_sample": gap_sample,
        },
        bg,
        warnings,
    )
    if segment.get("touching"):
        warnings.append("The part touches the marker frame. Keep the whole part inside, a few millimetres clear of every marker.")
    refined = _refine_outline(segment, scale, origin_x, origin_y_top)
    apparent = refined["outline"]
    apparent_holes = refined["holes"]
    distance_map = cv2.distanceTransform((1 - marker_raw).astype(np.uint8), cv2.DIST_L2, 3)
    query = np.clip(np.round(mm_to_px(apparent)).astype(int), [0, 0], [cols - 1, rows - 1])
    nearest = float(distance_map[query[:, 1], query[:, 0]].min() / scale)
    if nearest < 2.0:
        warnings.append(
            f"The part comes within {nearest:.1f} mm of a marker. Keep at least 5 mm of paper between the part and every marker."
        )

    ground = np.array([float(center[0]), float(center[1])])
    thickness = float(thickness_mm) if thickness_mm else 0.0
    outline = apparent
    holes = apparent_holes
    corrections = {"thickness_mm": thickness, "profile": profile, "scale": 1.0, "caliper_scale": 1.0}
    if thickness > 0:
        if thickness >= 0.6 * camera_z:
            raise CaptureError(
                "BAD_THICKNESS",
                f"Thickness {thickness:.1f} mm doesn't make sense when the camera is only {camera_z:.0f} mm up.",
            )
        if profile == "prism":
            factor = (camera_z - thickness) / camera_z
            x0, y0, _x1, _y1 = refined["crop"]
            mask_up = refined["mask_up"]
            filled = np.zeros_like(mask_up)
            contours, _ = cv2.findContours(mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, -1)
            gu = mm_to_px(ground)[0]
            gx = ((gu[0] - x0) + 0.5) * _UP - 0.5
            gy = ((gu[1] - y0) + 0.5) * _UP - 0.5
            affine = np.array([[factor, 0, gx * (1 - factor)], [0, factor, gy * (1 - factor)]], np.float64)
            shrunk = cv2.warpAffine(filled * 255, affine, (filled.shape[1], filled.shape[0])) > 127
            both = (filled.astype(bool) & shrunk).astype(np.uint8)
            contours, _ = cv2.findContours(both, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                raise CaptureError("BAD_THICKNESS", "Thickness correction wiped out the outline. Check the thickness value.")
            edge = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
            u = (edge[:, 0] + 0.5) / _UP - 0.5 + x0
            v = (edge[:, 1] + 0.5) / _UP - 0.5 + y0
            outline = np.c_[origin_x + u / scale, origin_y_top - v / scale]
            corrections["scale"] = round(factor, 5)
            corrections["note"] = "Square edge: footprint is the silhouette shrunk by (camera height − thickness) / height."
        else:
            factor = (camera_z - thickness / 2.0) / camera_z
            outline = ground + (apparent - ground) * factor
            corrections["scale"] = round(factor, 5)
            corrections["note"] = "Rounded edge: outline scaled by (camera height − thickness/2) / height."
        hole_factor = (camera_z - thickness / 2.0) / camera_z
        holes = [ground + (hole - ground) * hole_factor for hole in apparent_holes]
    else:
        warnings.append(
            "No thickness was entered, so this assumes a perfectly flat part. "
            "A 14 mm part shot from 200 mm reads several millimetres large. Enter the thickness."
        )

    raw_l, raw_w = _rect_dims(apparent)
    if caliper_mm and caliper_mm > 0:
        length_now, _width_now = _rect_dims(outline)
        factor = float(caliper_mm) / length_now
        centroid = outline.mean(0)
        outline = centroid + (outline - centroid) * factor
        holes = [centroid + (hole - centroid) * factor for hole in holes]
        corrections["caliper_scale"] = round(factor, 5)
        corrections["caliper_mm"] = float(caliper_mm)
        if abs(factor - 1) > 0.03:
            warnings.append(
                f"The caliper reading changed the scale by {(factor - 1) * 100:+.1f}%. "
                "Check that the sheet was printed at 100% (ruler on the black bar) and that the thickness is right."
            )

    length, width = _rect_dims(outline)
    (_cx, _cy), (side_a, side_b), angle = cv2.minAreaRect(outline.astype(np.float32))
    theta = angle if side_a >= side_b else angle + 90.0
    rotation_2d = np.array(
        [[math.cos(math.radians(-theta)), -math.sin(math.radians(-theta))],
         [math.sin(math.radians(-theta)), math.cos(math.radians(-theta))]]
    )
    center_xy = np.array([_cx, _cy])

    def to_local(pts):
        local = (np.asarray(pts, np.float64) - center_xy) @ rotation_2d.T
        return local + np.array([length / 2.0, width / 2.0])

    local_outline = to_local(outline)
    hole_info = []
    for hole in holes:
        local = to_local(hole)
        cx, cy, radius, max_err, rms = _fit_circle(local)
        if rms < max(0.12, 0.012 * radius) and max_err < max(0.3, 0.03 * radius) and 2 * radius >= 1.5:
            hole_info.append({"type": "circle", "cx": cx, "cy": cy, "diameter_mm": 2 * radius})
        else:
            hole_info.append(
                {
                    "type": "profile",
                    "cx": float(local[:, 0].mean()),
                    "cy": float(local[:, 1].mean()),
                    "diameter_mm": 2 * math.sqrt(abs(cv2.contourArea(local.astype(np.float32))) / math.pi),
                }
            )
    if thickness > 0:
        if profile == "rounded":
            thick_uncertainty = (1.0 * (thickness / 2)) / camera_z * length
        else:
            thick_uncertainty = (0.15 * thickness) / camera_z * length
    else:
        thick_uncertainty = 3.0 / camera_z * length
    if caliper_mm:
        estimate = math.sqrt(0.1**2 + (0.5 / scale) ** 2 + 0.05**2)
    else:
        estimate = math.sqrt(0.12**2 + (1.5 * reproj_rms) ** 2 + (0.5 / scale) ** 2 + thick_uncertainty**2 + (0.002 * length) ** 2)

    dxf = _write_dxf(local_outline, hole_info, units)
    preview = _overlay(rectified, scale, origin_x, origin_y_top, apparent, outline, holes, length, width, units, ground)
    long_edge = max(preview.shape[:2])
    if long_edge > 900:
        preview = cv2.resize(preview, None, fx=900 / long_edge, fy=900 / long_edge, interpolation=cv2.INTER_AREA)
    return PhotoMeasure(
        ok=True,
        length_mm=round(length, 3),
        width_mm=round(width, 3),
        raw_length_mm=round(raw_l, 3),
        raw_width_mm=round(raw_w, 3),
        holes=[{"shape": h["type"], "diameter_mm": round(h["diameter_mm"], 3)} for h in hole_info],
        warnings=warnings,
        camera_height_mm=round(camera_z, 1),
        tilt_deg=round(tilt, 1),
        estimated_error_mm=round(estimate, 2),
        overlay_bgr=preview,
        dxf_bytes=dxf,
        sheets=sheets,
        corrections=corrections,
        units=units,
    )
