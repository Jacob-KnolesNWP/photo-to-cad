"""Textured synthetic object for the 3D pipeline test.

A 100 × 70 × 40 mm box sits on the floor, with four printed markers around it.
Cameras walk around at two heights. The textures are stable across views so
feature matching has something to lock onto.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from partcapture.pipeline.sheets import get_dictionary, make_instance


def _texture(seed: int, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 255, np.uint8)
    # Lots of small blobs. A few big ones don't give SIFT enough corners,
    # and a single flat face is a degenerate scene for photogrammetry.
    for _ in range(160):
        center = rng.integers(0, size, size=2)
        axes = rng.integers(3, 16, size=2)
        color = tuple(int(v) for v in rng.integers(0, 255, size=3))
        cv2.ellipse(image, (int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])), 0, 0, 360, color, -1)
    cv2.rectangle(image, (6, 6), (size - 6, size - 6), (0, 0, 0), 3)
    return image


def _project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, k: np.ndarray) -> np.ndarray:
    camera = (rotation @ points.T).T + translation
    z = camera[:, 2:3]
    z_safe = np.where(np.abs(z) < 1e-8, 1e-8, z)
    uv = (camera / z_safe) @ k.T
    return np.column_stack([uv[:, 0], uv[:, 1], camera[:, 2]])


def _draw_textured_triangle(
    image: np.ndarray,
    zbuf: np.ndarray,
    tri_px: np.ndarray,
    tri_uv: np.ndarray,
    texture: np.ndarray,
) -> None:
    h, w = image.shape[:2]
    if np.any(tri_px[:, 2] <= 1.0):
        return
    min_x = max(0, int(np.floor(tri_px[:, 0].min())))
    max_x = min(w - 1, int(np.ceil(tri_px[:, 0].max())))
    min_y = max(0, int(np.floor(tri_px[:, 1].min())))
    max_y = min(h - 1, int(np.ceil(tri_px[:, 1].max())))
    if min_x >= max_x or min_y >= max_y:
        return
    xs = np.arange(min_x, max_x + 1)
    ys = np.arange(min_y, max_y + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    p0, p1, p2 = tri_px[:, :2]
    denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(denom) < 1e-8:
        return
    w0 = ((p1[1] - p2[1]) * (grid_x - p2[0]) + (p2[0] - p1[0]) * (grid_y - p2[1])) / denom
    w1 = ((p2[1] - p0[1]) * (grid_x - p2[0]) + (p0[0] - p2[0]) * (grid_y - p2[1])) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not np.any(inside):
        return
    # Perspective-correct depth and UVs. Screen-space interpolation makes the
    # texture swim between camera angles, and feature matching then fails.
    z0, z1, z2 = (float(tri_px[0, 2]), float(tri_px[1, 2]), float(tri_px[2, 2]))
    inv_z = w0 / z0 + w1 / z1 + w2 / z2
    z = 1.0 / inv_z
    closer = inside & (z < zbuf[min_y : max_y + 1, min_x : max_x + 1]) & (z > 1.0)
    if not np.any(closer):
        return
    u = (w0 * tri_uv[0, 0] / z0 + w1 * tri_uv[1, 0] / z1 + w2 * tri_uv[2, 0] / z2) / inv_z
    v = (w0 * tri_uv[0, 1] / z0 + w1 * tri_uv[1, 1] / z1 + w2 * tri_uv[2, 1] / z2) / inv_z
    th, tw = texture.shape[:2]
    tx = np.clip((u * (tw - 1)).astype(np.int32), 0, tw - 1)
    ty = np.clip((v * (th - 1)).astype(np.int32), 0, th - 1)
    patch = image[min_y : max_y + 1, min_x : max_x + 1]
    patch[closer] = texture[ty[closer], tx[closer]]
    zbuf[min_y : max_y + 1, min_x : max_x + 1][closer] = z[closer]


def _draw_quad(image, zbuf, world_corners, texture, rotation, translation, k) -> None:
    projected = _project(world_corners, rotation, translation, k)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    for indices in ((0, 1, 2), (0, 2, 3)):
        _draw_textured_triangle(
            image,
            zbuf,
            projected[list(indices)],
            uvs[list(indices)],
            texture,
        )


def _box_quads(
    length: float,
    width: float,
    z0: float,
    z1: float,
    cx: float = 0.0,
    cy: float = 0.0,
) -> list[np.ndarray]:
    x0, x1 = cx - length / 2, cx + length / 2
    y0, y1 = cy - width / 2, cy + width / 2
    return [
        np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),
        np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),
        np.array([[x1, y1, z0], [x0, y1, z0], [x0, y1, z1], [x1, y1, z1]]),
        np.array([[x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1]]),
        np.array([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]),
    ]


def _upright_marker(center: np.ndarray, right: np.ndarray, size: float) -> np.ndarray:
    """A marker standing up, so a camera walking around can read it.

    `right` is the marker's +X in the marker image (the viewer's right when
    looking at the printed face). Up is +Z.
    """
    right = np.asarray(right, dtype=np.float64)
    right = right / np.linalg.norm(right)
    up = np.array([0.0, 0.0, 1.0])
    half = size / 2.0
    center = np.asarray(center, dtype=np.float64)
    return np.stack(
        [
            center - right * half + up * half,  # TL
            center + right * half + up * half,  # TR
            center + right * half - up * half,  # BR
            center - right * half - up * half,  # BL
        ]
    )


def sample_object_views(
    n_views: int = 12,
    image_size: tuple[int, int] = (640, 480),
) -> tuple[list[np.ndarray], dict]:
    """Return photos and the ground-truth box size in millimetres."""
    # Two stacked blocks so the scene is not a single plane. Overall envelope
    # is 100 × 70 × 50 mm (the upper block sits inside the base footprint).
    length, width, height = 100.0, 70.0, 50.0
    width_px, height_px = image_size
    focal = 1.2 * max(width_px, height_px)
    k = np.array([[focal, 0, (width_px - 1) / 2], [0, focal, (height_px - 1) / 2], [0, 0, 1]], dtype=np.float64)
    textures = [_texture(100 + i) for i in range(10)]
    quads = _box_quads(length, width, 0.0, 22.0) + _box_quads(48.0, 36.0, 22.0, height, cx=8.0, cy=-4.0)
    floor = np.array([[-180, -150, 0], [180, -150, 0], [180, 150, 0], [-180, 150, 0]], dtype=np.float64)
    floor_tex = np.full((64, 64, 3), 235, np.uint8)
    cv2.line(floor_tex, (0, 32), (63, 32), (180, 180, 180), 1)

    # Large-format markers (36 mm) stay readable from a walk-around distance.
    sheet = make_instance("large", 0)
    marker_specs = list(sheet.markers_by_id.items())[:4]
    # Stand the markers up around the block. Flat on the floor they turn into
    # thin slivers from a walking camera and the detector cannot read them.
    placements = [
        (np.array([0.0, -95.0, 28.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([0.0, 95.0, 28.0]), np.array([-1.0, 0.0, 0.0])),
        (np.array([-110.0, 0.0, 28.0]), np.array([0.0, -1.0, 0.0])),
        (np.array([110.0, 0.0, 28.0]), np.array([0.0, 1.0, 0.0])),
    ]
    dictionary = get_dictionary()
    marker_textures = []
    for (marker_id, spec), (center, right) in zip(marker_specs, placements):
        gray = cv2.aruco.generateImageMarker(dictionary, int(marker_id), 160)
        marker_textures.append((_upright_marker(center, right, spec.size_mm), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

    images = []
    for index in range(n_views):
        angle = 2 * math.pi * index / n_views
        # Small steps. SIFT stops matching once the camera has swung much past 30°.
        elev = math.radians(42)
        radius = 300.0
        eye = np.array(
            [
                radius * math.cos(elev) * math.cos(angle),
                radius * math.cos(elev) * math.sin(angle),
                radius * math.sin(elev) + 30.0,
            ]
        )
        target = np.array([0.0, 0.0, height * 0.45])
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward], axis=0)
        translation = -rotation @ eye
        image = np.full((height_px, width_px, 3), 250, np.uint8)
        zbuf = np.full((height_px, width_px), np.inf)
        _draw_quad(image, zbuf, floor, floor_tex, rotation, translation, k)
        for quad, texture in zip(quads, textures):
            _draw_quad(image, zbuf, quad, texture, rotation, translation, k)
        for quad, texture in marker_textures:
            _draw_quad(image, zbuf, quad, texture, rotation, translation, k)
        images.append(image)
    truth = {
        "length_mm": length,
        "width_mm": width,
        "height_mm": height,
        "marker_count": len(marker_textures),
        "marker_size_mm": marker_specs[0][1].size_mm,
    }
    return images, truth
