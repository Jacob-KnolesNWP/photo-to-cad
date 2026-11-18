"""Synthetic flat scenes for tests and the in-app sample.

The virtual camera is a pinhole looking at the marker plane (Z = 0). A
homography is the exact model for that photo when distortion is zero, so the
accuracy tests measure detection, segmentation, and geometry fitting rather
than an arbitrary warp.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from partcapture.pipeline.mosaic import Pose2D
from partcapture.pipeline.sheets import SheetInstance, get_dictionary, make_instance


def work_area(layout_key: str, sheet_index: int = 0) -> tuple[float, float, float, float]:
    """Clear interior of a frame sheet, in that sheet's local millimetres."""
    inst = make_instance(layout_key, sheet_index)
    markers = list(inst.markers_by_id.values())
    size = markers[0].size_mm
    left = min(m.x_mm for m in markers) + size / 2 + 4
    right = max(m.x_mm for m in markers) - size / 2 - 4
    bottom = min(m.y_mm for m in markers) + size / 2 + 4
    top = max(m.y_mm for m in markers) - size / 2 - 4
    return left, bottom, right, top


def render_plane(
    sheets: list[tuple[SheetInstance, Pose2D]],
    parts: list[dict],
    px_per_mm: float = 8.0,
    margin_mm: float = 15.0,
) -> dict:
    """Rasterize sheets and parts on Z = 0.

    parts: {outer: Nx2 world mm, holes: [Nx2], color: (b,g,r)}
    """
    points = []
    for inst, pose in sheets:
        corners = np.array(
            [
                [0.0, 0.0],
                [inst.page_width_mm, 0.0],
                [inst.page_width_mm, inst.page_height_mm],
                [0.0, inst.page_height_mm],
            ]
        )
        points.append(pose.apply(corners))
    for part in parts:
        points.append(np.asarray(part["outer"], dtype=np.float64))
    all_pts = np.vstack(points)
    origin_x = float(all_pts[:, 0].min() - margin_mm)
    origin_y = float(all_pts[:, 1].min() - margin_mm)
    width_mm = float(all_pts[:, 0].max() + margin_mm - origin_x)
    height_mm = float(all_pts[:, 1].max() + margin_mm - origin_y)
    cols = int(math.ceil(width_mm * px_per_mm))
    rows = int(math.ceil(height_mm * px_per_mm))
    image = np.full((rows, cols, 3), 255, np.uint8)

    def world_to_px(world_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
        # OpenCV's integer pixel coordinate is the pixel center. The detectable
        # outer corner of a marker raster sits half a pixel outside the pasted
        # image's first pixel, so destinations use this same center convention.
        u = (pts[:, 0] - origin_x) * px_per_mm
        v = (origin_y + height_mm - pts[:, 1]) * px_per_mm
        return np.column_stack([u, v]).astype(np.float32)

    dictionary = get_dictionary()
    for inst, pose in sheets:
        for marker_id, spec in inst.markers_by_id.items():
            corners = pose.apply(spec.corners_local())
            side = max(20, int(round(spec.size_mm * px_per_mm)))
            marker = cv2.aruco.generateImageMarker(dictionary, int(marker_id), side)
            marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            # Detectable corners lie on the outer boundary, ~0.5 px outside the
            # first/last pixel centers of the generated image.
            src = np.array(
                [
                    [-0.5, -0.5],
                    [side - 0.5, -0.5],
                    [side - 0.5, side - 0.5],
                    [-0.5, side - 0.5],
                ],
                dtype=np.float32,
            )
            dst = world_to_px(corners)
            homography = cv2.getPerspectiveTransform(src, dst)
            cv2.warpPerspective(
                marker_bgr,
                homography,
                (cols, rows),
                dst=image,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_TRANSPARENT,
            )
    for part in parts:
        outer = world_to_px(part["outer"]).reshape(-1, 1, 2)
        cv2.fillPoly(image, [np.round(outer).astype(np.int32)], part.get("color", (40, 70, 140)))
        for hole in part.get("holes", []):
            hpx = world_to_px(hole).reshape(-1, 1, 2)
            cv2.fillPoly(image, [np.round(hpx).astype(np.int32)], (255, 255, 255))
    return {
        "image": image,
        "origin_mm": (origin_x, origin_y),
        "width_mm": width_mm,
        "height_mm": height_mm,
        "px_per_mm": px_per_mm,
    }


def look_at(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera pose as X_cam = R @ X_world + t, OpenCV convention (Y down, Z forward)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0])
    # If we're looking nearly straight down, world +Y is a fine image-up hint.
    if abs(np.dot(forward, up)) > 0.95:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, -up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ eye
    return rotation, translation


def render_photo(
    plane: dict,
    eye: np.ndarray,
    target: np.ndarray,
    image_size: tuple[int, int] = (1200, 900),
    focal_scale: float = 1.15,
    distortion_k1: float = 0.0,
) -> np.ndarray:
    """Render a pinhole (optional one-parameter radial distortion) view of the plane."""
    width, height = image_size
    rotation, translation = look_at(np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64))
    focal = focal_scale * max(width, height)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    k_inv = np.array([[1 / focal, 0, -cx / focal], [0, 1 / focal, -cy / focal], [0, 0, 1]], dtype=np.float64)
    us, vs = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    pix = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3)
    dirs_cam = pix @ k_inv.T
    if abs(distortion_k1) > 1e-8:
        # Pixel is the distorted observation. Undo the same model the solver uses
        # (normalized radius, factor 1 + k1 r^2) so a non-zero k1 is recoverable.
        scale = 0.5 * max(width, height)
        x = (pix[:, 0] - cx) / scale
        y = (pix[:, 1] - cy) / scale
        xu, yu = x.copy(), y.copy()
        for _ in range(8):
            r2 = xu * xu + yu * yu
            factor = 1.0 + distortion_k1 * r2
            factor = np.where(np.abs(factor) < 1e-4, 1e-4, factor)
            xu = x / factor
            yu = y / factor
        ideal = np.column_stack([cx + xu * scale, cy + yu * scale, np.ones_like(xu)])
        dirs_cam = ideal @ k_inv.T
    rot_t = rotation.T
    camera_center = -rot_t @ translation
    dirs_world = dirs_cam @ rot_t.T
    dz = dirs_world[:, 2]
    dz = np.where(np.abs(dz) < 1e-9, 1e-9, dz)
    depth = -camera_center[2] / dz
    hit = camera_center + depth[:, None] * dirs_world
    origin_x, origin_y = plane["origin_mm"]
    px = plane["px_per_mm"]
    plane_h = plane["height_mm"]
    map_x = ((hit[:, 0] - origin_x) * px).reshape(height, width).astype(np.float32)
    map_y = ((origin_y + plane_h - hit[:, 1]) * px).reshape(height, width).astype(np.float32)
    # Rays that miss the plane (behind the camera) get a white background.
    valid = (depth > 0).reshape(height, width)
    photo = cv2.remap(
        plane["image"],
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    photo[~valid] = 255
    return photo


def rectangle(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def circle(cx: float, cy: float, radius: float, n: int = 96) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])


def obround(cx: float, cy: float, length: float, width: float, n_cap: int = 48) -> np.ndarray:
    """Stadium shape, length along X, centered on (cx, cy)."""
    radius = width / 2.0
    straight = max(0.0, length / 2.0 - radius)
    left = []
    # Left cap, from top to bottom (CCW overall if we then go along the bottom to the right cap).
    for i in range(n_cap + 1):
        a = math.pi / 2 + math.pi * i / n_cap
        left.append([cx - straight + radius * math.cos(a), cy + radius * math.sin(a)])
    right = []
    for i in range(n_cap + 1):
        a = -math.pi / 2 + math.pi * i / n_cap
        right.append([cx + straight + radius * math.cos(a), cy + radius * math.sin(a)])
    return np.array(left + right, dtype=np.float64)


def sample_flat_plate() -> tuple[list[np.ndarray], dict]:
    """A 100 × 60 mm plate with a 16 mm hole, one letter sheet, one tilted photo."""
    inst = make_instance("letter", 0)
    left, bottom, right, top = work_area("letter", 0)
    cx = (left + right) / 2
    cy = (bottom + top) / 2
    outer = rectangle(cx - 50, cy - 30, cx + 50, cy + 30)
    hole = circle(cx, cy, 8.0)
    pose = Pose2D()
    plane = render_plane([(inst, pose)], [{"outer": outer, "holes": [hole], "color": (45, 85, 150)}], px_per_mm=8)
    eye = np.array([cx, cy - 40, 520.0])
    target = np.array([cx, cy, 0.0])
    photo = render_photo(plane, eye, target, image_size=(1100, 1400), focal_scale=1.05)
    truth = {"length_mm": 100.0, "width_mm": 60.0, "hole_diameter_mm": 16.0, "layout": "letter"}
    return [photo], truth
