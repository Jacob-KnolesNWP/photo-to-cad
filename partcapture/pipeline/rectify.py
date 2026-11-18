"""Plane mapping from a photo: homography, optional radial distortion, RMS."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from partcapture.pipeline.errors import CaptureError


@dataclass
class PlaneMapping:
    homography: np.ndarray  # 3x3, world mm -> ideal pixels
    k1: float
    rms_px: float
    rms_mm: float
    inlier_ratio: float
    px_per_mm: float
    image_shape: tuple[int, int]  # h, w

    def project(self, world_xy: np.ndarray) -> np.ndarray:
        return project_world(world_xy, self.homography, self.k1, self.image_shape)

    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        return unproject_pixels(pixels, self.homography, self.k1, self.image_shape)


def _center_scale(image_shape: tuple[int, ...]) -> tuple[float, float, float]:
    h, w = int(image_shape[0]), int(image_shape[1])
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    scale = 0.5 * max(w, h)
    return cx, cy, scale


def _distort(ideal_px: np.ndarray, k1: float, image_shape: tuple[int, ...]) -> np.ndarray:
    if abs(k1) < 1e-12:
        return ideal_px
    cx, cy, scale = _center_scale(image_shape)
    x = (ideal_px[:, 0] - cx) / scale
    y = (ideal_px[:, 1] - cy) / scale
    r2 = x * x + y * y
    factor = 1.0 + k1 * r2
    return np.column_stack([cx + x * factor * scale, cy + y * factor * scale])


def _undistort(obs_px: np.ndarray, k1: float, image_shape: tuple[int, ...], iters: int = 8) -> np.ndarray:
    if abs(k1) < 1e-12:
        return obs_px
    cx, cy, scale = _center_scale(image_shape)
    x = (obs_px[:, 0] - cx) / scale
    y = (obs_px[:, 1] - cy) / scale
    xu = x.copy()
    yu = y.copy()
    for _ in range(iters):
        r2 = xu * xu + yu * yu
        factor = 1.0 + k1 * r2
        # Avoid blow-ups if k1 * r2 ≈ -1.
        factor = np.where(np.abs(factor) < 1e-4, 1e-4, factor)
        xu = x / factor
        yu = y / factor
    return np.column_stack([cx + xu * scale, cy + yu * scale])


def project_world(world_xy: np.ndarray, homography: np.ndarray, k1: float, image_shape: tuple[int, ...]) -> np.ndarray:
    pts = np.asarray(world_xy, dtype=np.float64).reshape(-1, 1, 2)
    ideal = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
    return _distort(ideal, k1, image_shape)


def unproject_pixels(pixels: np.ndarray, homography: np.ndarray, k1: float, image_shape: tuple[int, ...]) -> np.ndarray:
    ideal = _undistort(np.asarray(pixels, dtype=np.float64).reshape(-1, 2), k1, image_shape)
    inv = np.linalg.inv(homography)
    world = cv2.perspectiveTransform(ideal.reshape(-1, 1, 2), inv).reshape(-1, 2)
    return world


def _rms(world_xy: np.ndarray, pixels: np.ndarray, homography: np.ndarray, k1: float, image_shape) -> tuple[float, float]:
    pred = project_world(world_xy, homography, k1, image_shape)
    delta = pred - pixels
    per = np.linalg.norm(delta, axis=1)
    rms_px = float(np.sqrt(np.mean(per**2)))
    # Local millimetres per pixel from the homography at the cloud centroid.
    px_per_mm = _local_px_per_mm(world_xy, homography, k1, image_shape)
    rms_mm = rms_px / px_per_mm if px_per_mm > 1e-6 else 1e9
    return rms_px, rms_mm


def _local_px_per_mm(world_xy: np.ndarray, homography: np.ndarray, k1: float, image_shape) -> float:
    center = np.asarray(world_xy, dtype=np.float64).mean(axis=0)
    samples = np.vstack(
        [
            center,
            center + np.array([1.0, 0.0]),
            center + np.array([0.0, 1.0]),
        ]
    )
    proj = project_world(samples, homography, k1, image_shape)
    dx = np.linalg.norm(proj[1] - proj[0])
    dy = np.linalg.norm(proj[2] - proj[0])
    return float(max(1e-6, 0.5 * (dx + dy)))


def _fit_k1(world_xy: np.ndarray, pixels: np.ndarray, h_init: np.ndarray, image_shape) -> tuple[np.ndarray, float, float]:
    h, w = int(image_shape[0]), int(image_shape[1])

    def unpack(p: np.ndarray) -> tuple[np.ndarray, float]:
        homography = np.array(
            [[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]],
            dtype=np.float64,
        )
        return homography, float(p[8])

    x0 = np.concatenate([h_init.ravel()[:8], [0.0]])

    def residuals(p: np.ndarray) -> np.ndarray:
        homography, k1 = unpack(p)
        pred = project_world(world_xy, homography, k1, (h, w))
        return (pred - pixels).ravel()

    try:
        result = least_squares(residuals, x0, method="lm", max_nfev=60)
    except Exception:
        return h_init, 0.0, 1e9
    homography, k1 = unpack(result.x)
    if not np.isfinite(homography).all() or not np.isfinite(k1) or abs(k1) > 0.8:
        return h_init, 0.0, 1e9
    rms_px, _rms_mm = _rms(world_xy, pixels, homography, k1, (h, w))
    return homography, k1, rms_px


def estimate_mapping(world_xy: np.ndarray, pixels: np.ndarray, image_shape: tuple[int, ...]) -> PlaneMapping:
    world_xy = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if len(world_xy) < 4:
        raise CaptureError(
            "TOO_FEW_MARKERS",
            "Not enough marker corners were readable to straighten this photo. "
            "Keep at least four markers fully in the frame.",
        )
    homography, mask = cv2.findHomography(world_xy, pixels, cv2.RANSAC, 3.0)
    if homography is None:
        raise CaptureError(
            "MARKERS_NOT_PLANAR",
            "The markers don't line up on a flat plane. Check that the sheets are lying flat "
            "and that more than one row of markers is visible.",
        )
    if mask is None:
        inlier_ratio = 1.0
    else:
        inlier_ratio = float(mask.ravel().mean())
    rms0, rms0_mm = _rms(world_xy, pixels, homography, 0.0, image_shape)
    h_k, k1, rms_k = _fit_k1(world_xy, pixels, homography, image_shape)
    # Only keep the distortion term when it clearly beats a plain homography.
    # On a clean pinhole photo of a plane the homography is already exact, and
    # a free k1 can trade against it and make the result worse.
    if rms_k < rms0 * 0.85 and abs(k1) >= 1e-4:
        homography = h_k
        used_k = k1
        rms_px = rms_k
    else:
        used_k = 0.0
        rms_px = rms0
    _px, rms_mm = _rms(world_xy, pixels, homography, used_k, image_shape)
    px_per_mm = _local_px_per_mm(world_xy, homography, used_k, image_shape)
    if not np.isfinite(rms_mm):
        rms_mm = rms0_mm
    return PlaneMapping(
        homography=homography,
        k1=float(used_k),
        rms_px=float(rms_px),
        rms_mm=float(rms_mm),
        inlier_ratio=inlier_ratio,
        px_per_mm=float(px_per_mm),
        image_shape=(int(image_shape[0]), int(image_shape[1])),
    )


def footprint_mm(mapping: PlaneMapping, samples: int = 8) -> np.ndarray:
    """World polygon of the photo's visible plane, from the image border."""
    h, w = mapping.image_shape
    xs = np.linspace(0, w - 1, samples)
    ys = np.linspace(0, h - 1, samples)
    border = []
    border.extend([(x, 0) for x in xs])
    border.extend([(w - 1, y) for y in ys])
    border.extend([(x, h - 1) for x in xs[::-1]])
    border.extend([(0, y) for y in ys[::-1]])
    return mapping.unproject(np.asarray(border, dtype=np.float64))
