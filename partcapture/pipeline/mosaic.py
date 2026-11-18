"""Solve sheet poses on the table and stitch photos into one scaled plane."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from partcapture.pipeline.detect import MarkerObservation
from partcapture.pipeline.errors import CaptureError
from partcapture.pipeline.rectify import PlaneMapping, estimate_mapping
from partcapture.pipeline.sheets import SheetInstance, make_instance


@dataclass
class Pose2D:
    theta: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    def rotation(self) -> np.ndarray:
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return pts @ self.rotation().T + np.array([self.tx, self.ty])

    def compose_after(self, child: "Pose2D") -> "Pose2D":
        """Return pose of a child expressed in this pose's parent frame.

        child maps child-local -> this local. Result maps child-local -> parent.
        """
        rot = self.rotation() @ child.rotation()
        theta = math.atan2(rot[1, 0], rot[0, 0])
        translation = self.rotation() @ np.array([child.tx, child.ty]) + np.array([self.tx, self.ty])
        return Pose2D(theta, float(translation[0]), float(translation[1]))


def rigid_align(src: np.ndarray, dst: np.ndarray) -> Pose2D:
    """Least-squares rigid transform taking src points onto dst (SE(2))."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    cov = (src - cs).T @ (dst - cd)
    u, _s, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt = vt.copy()
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    translation = cd - rot @ cs
    theta = math.atan2(rot[1, 0], rot[0, 0])
    return Pose2D(theta, float(translation[0]), float(translation[1]))


@dataclass
class View:
    index: int  # 0-based photo index
    image: np.ndarray
    observations: list[MarkerObservation]
    sharpness: float
    mapping: PlaneMapping | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class Mosaic:
    image: np.ndarray
    valid: np.ndarray  # bool
    origin_mm: tuple[float, float]  # min x, min y (y-up)
    px_per_mm: float
    width_mm: float
    height_mm: float
    poses: dict[tuple[str, int], Pose2D]
    views: list[View]
    instances: dict[tuple[str, int], SheetInstance]


def _sheet_key(obs: MarkerObservation) -> tuple[str, int]:
    return (obs.layout_key, obs.sheet_index)


def _group(observations: list[MarkerObservation]) -> dict[tuple[str, int], list[MarkerObservation]]:
    grouped: dict[tuple[str, int], list[MarkerObservation]] = {}
    for obs in observations:
        grouped.setdefault(_sheet_key(obs), []).append(obs)
    return grouped


def _mapping_from_sheet(
    observations: list[MarkerObservation], image_shape: tuple[int, int]
) -> PlaneMapping:
    world = np.vstack([obs.corners_local_mm for obs in observations])
    pixels = np.vstack([obs.corners_px for obs in observations])
    return estimate_mapping(world, pixels, image_shape)


def _relative_edges(view: View) -> list[tuple[tuple[str, int], tuple[str, int], Pose2D, int]]:
    """Edges (anchor, other, pose of other in anchor frame, weight)."""
    grouped = _group(view.observations)
    anchors = [key for key, obs in grouped.items() if len(obs) >= 2]
    if not anchors:
        return []
    anchor = max(anchors, key=lambda key: len(grouped[key]))
    try:
        mapping = _mapping_from_sheet(grouped[anchor], view.image.shape)
    except CaptureError:
        return []
    edges = []
    for key, obs_list in grouped.items():
        if key == anchor:
            continue
        local = np.vstack([obs.corners_local_mm for obs in obs_list])
        pixels = np.vstack([obs.corners_px for obs in obs_list])
        in_anchor = mapping.unproject(pixels)
        if len(local) < 4:
            continue
        pose = rigid_align(local, in_anchor)
        pred = pose.apply(local)
        resid = float(np.mean(np.linalg.norm(pred - in_anchor, axis=1)))
        if resid > 5.0:
            view.warnings.append(
                f"Photo {view.index + 1} sees sheet {_label(key)} tilted or curled relative to "
                f"sheet {_label(anchor)} (mismatch {resid:.1f} mm). That link was ignored."
            )
            continue
        edges.append((anchor, key, pose, len(local)))
    return edges


def _label(key: tuple[str, int]) -> str:
    layout, index = key
    return f"{layout} {index + 1}"


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def solve_poses(views: list[View]) -> dict[tuple[str, int], Pose2D]:
    counts: dict[tuple[str, int], int] = {}
    for view in views:
        for obs in view.observations:
            counts[_sheet_key(obs)] = counts.get(_sheet_key(obs), 0) + 1
    if not counts:
        raise CaptureError(
            "NO_MARKERS",
            "No calibration markers were found. Print a marker sheet, lay the part on it, "
            "and photograph the sheet from above.",
        )
    edges = []
    for view in views:
        edges.extend(_relative_edges(view))

    # Union-find over sheets that were observed.
    parent = {key: key for key in counts}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for anchor, other, _pose, _w in edges:
        if anchor in parent and other in parent:
            union(anchor, other)

    # Sheets with no edge still form their own component. A single component
    # that was never linked to another is fine — it's one sheet, or several
    # copies of the same sheet id (which we don't allow). Multiple components
    # means the photos never showed neighboring sheets together.
    roots = {find(key) for key in counts}
    if len(roots) > 1:
        parts = []
        for root in roots:
            members = [ _label(key) for key in counts if find(key) == root]
            parts.append(", ".join(sorted(members)))
        raise CaptureError(
            "SHEETS_NOT_LINKED",
            "These photos don't tie every marker sheet into one measurement. "
            "The separate groups are: "
            + " | ".join(parts)
            + ". Retake so at least one photo shows markers from two neighboring sheets "
            "at the same time, with two markers on one of them.",
        )

    root = max(counts, key=lambda key: counts[key])
    poses: dict[tuple[str, int], Pose2D] = {root: Pose2D()}

    # BFS initialization along edges, in either direction.
    unused = list(edges)
    progressed = True
    while progressed:
        progressed = False
        still = []
        for anchor, other, pose_other_in_anchor, _w in unused:
            if anchor in poses and other not in poses:
                poses[other] = poses[anchor].compose_after(pose_other_in_anchor)
                progressed = True
            elif other in poses and anchor not in poses:
                # Invert: pose_other = pose_anchor ∘ relative => pose_anchor = pose_other ∘ inv(relative)
                inv = _invert_pose(pose_other_in_anchor)
                poses[anchor] = poses[other].compose_after(inv)
                progressed = True
            else:
                still.append((anchor, other, pose_other_in_anchor, _w))
        unused = still

    missing = [key for key in counts if key not in poses]
    if missing:
        names = ", ".join(_label(key) for key in missing)
        raise CaptureError(
            "SHEETS_NOT_LINKED",
            f"Couldn't place {names} on the same plane as the other sheets. "
            "Take an overlapping photo that includes that sheet next to one you already captured, "
            "with at least two markers from one sheet visible.",
        )

    if len(poses) == 1 or not edges:
        return poses

    keys = [key for key in poses if key != root]
    index_of = {key: i for i, key in enumerate(keys)}

    def pack(values: np.ndarray) -> dict[tuple[str, int], Pose2D]:
        solved = {root: Pose2D()}
        for key, i in index_of.items():
            solved[key] = Pose2D(float(values[3 * i]), float(values[3 * i + 1]), float(values[3 * i + 2]))
        return solved

    x0 = np.zeros(3 * len(keys), dtype=np.float64)
    for key, i in index_of.items():
        pose = poses[key]
        x0[3 * i : 3 * i + 3] = (pose.theta, pose.tx, pose.ty)

    def residuals(values: np.ndarray) -> np.ndarray:
        solved = pack(values)
        chunks = []
        for anchor, other, measured, weight in edges:
            pred = _pose_in_frame(solved[anchor], solved[other])
            # Compare measured (other in anchor) to predicted.
            dtheta = _wrap(pred.theta - measured.theta)
            dt = np.array([pred.tx - measured.tx, pred.ty - measured.ty])
            scale = math.sqrt(max(weight, 1))
            chunks.append(scale * np.array([dtheta * 40.0, dt[0], dt[1]]))  # 40 mm per radian ~ weighting
        if not chunks:
            return np.zeros(1)
        return np.concatenate(chunks)

    try:
        result = least_squares(residuals, x0, method="lm", max_nfev=80)
        if np.isfinite(result.x).all():
            return pack(result.x)
    except Exception:
        pass
    return poses


def _invert_pose(pose: Pose2D) -> Pose2D:
    rot_t = pose.rotation().T
    theta = math.atan2(rot_t[1, 0], rot_t[0, 0])
    translation = -(rot_t @ np.array([pose.tx, pose.ty]))
    return Pose2D(theta, float(translation[0]), float(translation[1]))


def _pose_in_frame(pose_anchor: Pose2D, pose_other: Pose2D) -> Pose2D:
    """Relative pose of other in the anchor's local frame."""
    inv_anchor = _invert_pose(pose_anchor)
    return inv_anchor.compose_after(pose_other)


def _instances_for(poses: dict[tuple[str, int], Pose2D]) -> dict[tuple[str, int], SheetInstance]:
    return {key: make_instance(key[0], key[1]) for key in poses}


def _choose_resolution(width_mm: float, height_mm: float) -> float:
    long_edge = max(width_mm, height_mm, 1.0)
    # 10 px/mm on a letter sheet (~280 mm) is 2800 px. Large parts step down
    # so the mosaic stays inside memory, and the reported tolerance grows.
    px = min(10.0, 4500.0 / long_edge)
    return float(max(2.0, px))


def _bounds(views: list[View], poses: dict[tuple[str, int], Pose2D]) -> tuple[float, float, float, float]:
    # The measurable plane is the sheets themselves, plus a margin so a part that
    # overhangs the paper a little is still inside the mosaic. Photo footprints are
    # not used: a tilted shot's horizon would blow the mosaic up to metres.
    del views
    points = []
    instances = _instances_for(poses)
    for key, pose in poses.items():
        inst = instances[key]
        corners = np.array(
            [
                [0.0, 0.0],
                [inst.page_width_mm, 0.0],
                [inst.page_width_mm, inst.page_height_mm],
                [0.0, inst.page_height_mm],
            ]
        )
        points.append(pose.apply(corners))
    all_pts = np.vstack(points)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    pad = 30.0
    return float(min_xy[0] - pad), float(min_xy[1] - pad), float(max_xy[0] + pad), float(max_xy[1] + pad)


def _assign_mappings(views: list[View], poses: dict[tuple[str, int], Pose2D]) -> None:
    for view in views:
        world = []
        pixels = []
        for obs in view.observations:
            key = _sheet_key(obs)
            if key not in poses:
                continue
            global_xy = poses[key].apply(obs.corners_local_mm)
            world.append(global_xy)
            pixels.append(obs.corners_px)
        if not world:
            view.warnings.append(
                f"Photo {view.index + 1} had no markers on a placed sheet and was skipped."
            )
            continue
        world_xy = np.vstack(world)
        pix = np.vstack(pixels)
        if len(world_xy) < 8:
            view.warnings.append(
                f"Photo {view.index + 1} only showed {len(world_xy) // 4} usable marker"
                f"{'' if len(world_xy) // 4 == 1 else 's'}. At least two fully visible markers "
                "are needed, so this photo was skipped."
            )
            continue
        try:
            view.mapping = estimate_mapping(world_xy, pix, view.image.shape)
        except CaptureError as exc:
            view.warnings.append(f"Photo {view.index + 1} was skipped: {exc.message}")


def build_mosaic(views: list[View]) -> Mosaic:
    if not views:
        raise CaptureError("NO_PHOTOS", "Add at least one photo of the part on the marker sheet.")
    poses = solve_poses(views)
    _assign_mappings(views, poses)
    usable = [view for view in views if view.mapping is not None]
    if not usable:
        detail = " ".join(w for view in views for w in view.warnings)
        raise CaptureError(
            "TOO_FEW_MARKERS",
            "None of the photos had enough readable markers to measure from. "
            "Keep at least four markers in each shot, in more than one row. " + detail,
        )
    # Reprojection sanity. A badly wrong photo should stop the job, not silently skew the part.
    worst = max(usable, key=lambda view: view.mapping.rms_mm)
    if worst.mapping.rms_mm > 1.5:
        raise CaptureError(
            "HIGH_REPROJECTION",
            f"Photo {worst.index + 1} doesn't agree with the markers "
            f"(about {worst.mapping.rms_mm:.2f} mm of mismatch). "
            "The sheet may be curled, the lens may be very close to the page, or a marker is damaged. "
            "Retake from a little farther away, with the sheet flat and evenly lit.",
        )

    min_x, min_y, max_x, max_y = _bounds(usable, poses)
    width_mm = max_x - min_x
    height_mm = max_y - min_y
    if width_mm < 20 or height_mm < 20 or width_mm > 8000 or height_mm > 8000:
        raise CaptureError(
            "BAD_EXTENT",
            "The measured area came out an implausible size. Retake the photos with the sheets flat "
            "and fully in frame.",
        )
    px_per_mm = _choose_resolution(width_mm, height_mm)
    cols = int(round(width_mm * px_per_mm))
    rows = int(round(height_mm * px_per_mm))
    cols = max(cols, 2)
    rows = max(rows, 2)

    mosaic = np.zeros((rows, cols, 3), dtype=np.float32)
    weight = np.zeros((rows, cols), dtype=np.float32)

    col_idx = np.arange(cols, dtype=np.float32)
    row_idx = np.arange(rows, dtype=np.float32)
    # Pixel centers in world millimetres, y up.
    world_x = min_x + (col_idx + 0.5) / px_per_mm
    world_y = min_y + height_mm - (row_idx + 0.5) / px_per_mm

    for view in usable:
        mapping = view.mapping
        assert mapping is not None
        # Sample a coarse grid first? Full remap is fine at our capped size.
        xx, yy = np.meshgrid(world_x, world_y)
        flat = np.column_stack([xx.ravel(), yy.ravel()])
        # Project in chunks to limit the transient array for the largest boards.
        src = np.empty((flat.shape[0], 2), dtype=np.float32)
        chunk = 250_000
        for start in range(0, flat.shape[0], chunk):
            block = flat[start : start + chunk]
            src[start : start + chunk] = mapping.project(block).astype(np.float32)
        map_x = src[:, 0].reshape(rows, cols)
        map_y = src[:, 1].reshape(rows, cols)
        h, w = view.image.shape[:2]
        inside = (map_x >= 1) & (map_x < w - 2) & (map_y >= 1) & (map_y < h - 2)
        if not inside.any():
            view.warnings.append(f"Photo {view.index + 1} did not overlap the measurement plane and was skipped.")
            continue
        warped = cv2.remap(
            view.image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        # Prefer pixels near the middle of the photo, where lens error is smaller.
        nx = (map_x - (w - 1) / 2.0) / max((w - 1) / 2.0, 1.0)
        ny = (map_y - (h - 1) / 2.0) / max((h - 1) / 2.0, 1.0)
        center_w = np.clip(1.0 - 0.85 * (nx * nx + ny * ny), 0.05, 1.0).astype(np.float32)
        center_w *= inside.astype(np.float32)
        better = center_w > weight
        if better.any():
            mosaic[better] = warped[better]
            weight[better] = center_w[better]

    valid = weight > 0
    if int(valid.sum()) < 100:
        raise CaptureError(
            "NO_COVERAGE",
            "The photos did not cover the marker sheets after straightening. "
            "Retake with the sheets filling more of the frame.",
        )
    image = np.clip(mosaic, 0, 255).astype(np.uint8)
    # Paint uncovered pixels white so segmentation treats them as "no photo", via the valid mask.
    image[~valid] = 255
    return Mosaic(
        image=image,
        valid=valid,
        origin_mm=(min_x, min_y),
        px_per_mm=px_per_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        poses=poses,
        views=views,
        instances=_instances_for(poses),
    )


def world_to_pixel(world_xy: np.ndarray, mosaic: Mosaic) -> np.ndarray:
    pts = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    origin_x, origin_y = mosaic.origin_mm
    u = (pts[:, 0] - origin_x) * mosaic.px_per_mm - 0.5
    v = (origin_y + mosaic.height_mm - pts[:, 1]) * mosaic.px_per_mm - 0.5
    return np.column_stack([u, v])


def pixel_to_world(pixels: np.ndarray, mosaic: Mosaic) -> np.ndarray:
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    origin_x, origin_y = mosaic.origin_mm
    x = origin_x + (pts[:, 0] + 0.5) / mosaic.px_per_mm
    y = origin_y + mosaic.height_mm - (pts[:, 1] + 0.5) / mosaic.px_per_mm
    return np.column_stack([x, y])
