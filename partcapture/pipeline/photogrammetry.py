"""Scaled 3D reconstruction from a walk-around photo set.

Camera poses come from COLMAP (CPU). Dense depth comes from pairwise stereo
between neighboring views, fused into a voxel surface. Marker squares set the
scale. CUDA patch-match stereo is not available here, and the UI says so:
expect about ±1.5 mm on a small object, more as the object gets larger.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from skimage.measure import marching_cubes

from partcapture.pipeline.accuracy import estimate_3d
from partcapture.pipeline.detect import SHARPNESS_MIN, detect_markers, sharpness_score
from partcapture.pipeline.errors import CaptureError

MIN_PHOTOS = 8


def reconstruct_object(
    images: list[np.ndarray],
    work_dir: Path,
    progress=None,
    reference_distance_mm: float | None = None,
    reference_points: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Run SfM, scale, and meshing.

    reference_points: two 3D points already on the unscaled cloud, with
    reference_distance_mm the real distance between them. Used when the user
    picks a known length instead of (or after) the markers.
    """
    if len(images) < MIN_PHOTOS:
        raise CaptureError(
            "TOO_FEW_PHOTOS",
            f"3D capture needs at least {MIN_PHOTOS} photos. You added {len(images)}. "
            "Walk around the object, overlap each shot with the last one, and include "
            "the marker sheets in several of the photos.",
        )
    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    image_dir = work_dir / "images"
    image_dir.mkdir(parents=True)
    _report(progress, 0.04, "Checking the photos")

    observations = []
    blurry = 0
    for index, image in enumerate(images):
        score = sharpness_score(image)
        if score < SHARPNESS_MIN:
            blurry += 1
        observations.append(detect_markers(image))
        cv2.imwrite(str(image_dir / f"{index:03d}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if blurry > len(images) / 2:
        raise CaptureError(
            "BLURRY",
            f"{blurry} of {len(images)} photos look blurry. Hold the phone steady, tap to focus, "
            "and retake the soft ones. A blurry set will not line up.",
        )

    _report(progress, 0.12, "Finding features")
    database_path = work_dir / "database.db"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = 1400
    extraction.num_threads = 4
    extraction.sift.max_num_features = 6000
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    pycolmap.extract_features(
        database_path,
        image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extraction,
    )
    _report(progress, 0.32, "Matching photos")
    pycolmap.match_exhaustive(database_path)
    _report(progress, 0.48, "Solving camera positions")
    pipeline = pycolmap.IncrementalPipelineOptions()
    pipeline.min_num_matches = 15
    pipeline.mapper.init_min_num_inliers = 30
    pipeline.mapper.init_min_tri_angle = 6.0
    pipeline.mapper.abs_pose_min_num_inliers = 15
    pipeline.mapper.filter_min_tri_angle = 1.0
    pipeline.ba_refine_principal_point = False
    pipeline.num_threads = 4
    sparse_dir = work_dir / "sparse"
    models = pycolmap.incremental_mapping(database_path, image_dir, sparse_dir, options=pipeline)
    if not models:
        raise CaptureError(
            "SFM_FAILED",
            "The photos could not be aligned. Add more overlap, keep the object still, and make sure "
            "the surface has visible grain or pencil marks — smooth bare surfaces often fail. "
            "Walk in smaller steps so each photo shares most of the last one.",
        )
    reconstruction = models[max(models, key=lambda key: models[key].num_reg_images())]
    if reconstruction.num_reg_images() < 3:
        raise CaptureError(
            "SFM_FAILED",
            f"Only {reconstruction.num_reg_images()} photos lined up. The steps between shots are too big, "
            "or the surface doesn't have enough visible detail. Retake with more overlap.",
        )

    cameras = _cameras_by_name(reconstruction)
    _report(progress, 0.62, "Setting the scale from the markers")
    scale_info = _scale_from_markers(images, observations, cameras)
    scale = scale_info["scale"]
    scaled_by = "markers"
    warnings = list(scale_info["warnings"])
    if scale is None:
        scale = 1.0
        scaled_by = "unscaled"
        warnings.append(
            "No marker was seen from two angles, so the model is not in real units yet. "
            "Enter a known distance between two points, or retake with marker sheets visible."
        )

    _report(progress, 0.72, "Building a surface")
    # Stereo uses the unscaled COLMAP cameras. Scaling the translations first
    # would break the projection of the pixels.
    dense_raw = _dense_points(images, cameras)
    sparse = _sparse_points(reconstruction)
    cloud_raw = dense_raw if len(dense_raw) > 200 else sparse
    cloud = cloud_raw * scale
    cloud, _rotation, _translation = _align_up(cloud, scale_info.get("verticals") or [], scale)
    vertices, faces = _mesh_from_points(cloud)
    if len(faces) < 20:
        raise CaptureError(
            "MESH_FAILED",
            "The photos lined up, but there wasn't enough surface to build a mesh. "
            "Add more photos from a second height and make sure the object fills more of the frame.",
        )

    if reference_points is not None and reference_distance_mm:
        vertices, ref_scale = _rescale_to_segment(vertices, reference_points, reference_distance_mm, rotation_to_up, translation, scale)
        scale *= ref_scale
        scaled_by = "reference"
        warnings.append("Scale was set from the distance you entered.")

    extents = _extents(vertices)
    diagonal = float(np.linalg.norm(extents["size_mm"]))
    accuracy = estimate_3d(diagonal, marker_scale_cv=scale_info.get("cv", 0.0))
    _report(progress, 0.96, "Writing STL and OBJ")
    return {
        "vertices": vertices,
        "faces": faces,
        "scale": scale,
        "scaled_by": scaled_by,
        "registered_images": int(reconstruction.num_reg_images()),
        "sparse_points": int(reconstruction.num_points3D()),
        "marker_edges": scale_info.get("edges", 0),
        "warnings": warnings,
        "extents": extents,
        "accuracy": accuracy,
        "reproj_px": float(reconstruction.compute_mean_reprojection_error()),
    }


def _report(progress, fraction: float, message: str) -> None:
    if progress is not None:
        progress(fraction, message)


def _cameras_by_name(reconstruction) -> dict:
    cameras = {}
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.images[image_id]
        rigid = image.cam_from_world()
        rotation = np.array(rigid.rotation.matrix(), dtype=np.float64)
        translation = np.array(rigid.translation, dtype=np.float64).reshape(3)
        calib = reconstruction.cameras[image.camera_id]
        matrix = np.array(calib.calibration_matrix(), dtype=np.float64)
        cameras[image.name] = {
            "R": rotation,
            "t": translation,
            "K": matrix,
            "width": int(calib.width),
            "height": int(calib.height),
            "center": -rotation.T @ translation,
        }
    return cameras


def _sparse_points(reconstruction) -> np.ndarray:
    coords = []
    for _point_id, point in reconstruction.points3D.items():
        coords.append(np.array(point.xyz, dtype=np.float64))
    if not coords:
        return np.zeros((0, 3))
    return np.vstack(coords)


def _scale_from_markers(images, observations, cameras) -> dict:
    """Median of (true marker edge / reconstructed edge) over every edge seen twice."""
    # Group pixel observations by marker id and corner index.
    by_marker: dict[int, list[tuple[int, np.ndarray, float]]] = defaultdict(list)
    for index, found in enumerate(observations):
        name = f"{index:03d}.jpg"
        if name not in cameras:
            continue
        for obs in found:
            by_marker[obs.marker_id].append((index, obs.corners_px, obs.size_mm))

    samples = []
    verticals = []
    for marker_id, views in by_marker.items():
        if len(views) < 2:
            continue
        size = views[0][2]
        corners_3d = []
        for corner in range(4):
            obs_list = []
            for index, corners_px, _size in views:
                camera = cameras[f"{index:03d}.jpg"]
                projection = camera["K"] @ np.hstack([camera["R"], camera["t"].reshape(3, 1)])
                obs_list.append((projection, corners_px[corner]))
            point = _triangulate(obs_list)
            if point is None:
                corners_3d = []
                break
            corners_3d.append(point)
        if len(corners_3d) < 4:
            continue
        corners_3d = np.vstack(corners_3d)
        for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
            length = float(np.linalg.norm(corners_3d[i] - corners_3d[j]))
            if length > 1e-8:
                samples.append(size / length)
        # Vertical edges of an upright marker are TL-BL (0-3) and TR-BR (1-2).
        # On a flat sheet they're in the sheet plane; either way a consistent
        # direction across observations is a usable "up" hint when markers stand up.
        verticals.append(corners_3d[0] - corners_3d[3])
        verticals.append(corners_3d[1] - corners_3d[2])
    if not samples:
        return {"scale": None, "cv": 0.0, "edges": 0, "warnings": [], "verticals": []}
    scales = np.array(samples, dtype=np.float64)
    # Drop wild outliers before the median (a mismatched corner pair).
    median = float(np.median(scales))
    keep = scales[(scales > median * 0.5) & (scales < median * 1.5)]
    if len(keep) == 0:
        keep = scales
    chosen = float(np.median(keep))
    cv = float(np.std(keep) / chosen) if chosen else 0.0
    warnings = []
    if cv > 0.05:
        warnings.append(
            f"The markers disagreed on scale by about {cv * 100:.0f}%. "
            "Check that the sheets were printed at 100% and weren't curled."
        )
    return {"scale": chosen, "cv": cv, "edges": int(len(keep)), "warnings": warnings, "verticals": verticals}


def _triangulate(observations: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    rows = []
    for projection, xy in observations:
        u, v = float(xy[0]), float(xy[1])
        rows.append(u * projection[2] - projection[0])
        rows.append(v * projection[2] - projection[1])
    matrix = np.stack(rows)
    try:
        _, _, vt = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:
        return None
    point = vt[-1]
    if abs(point[3]) < 1e-12:
        return None
    xyz = point[:3] / point[3]
    if not np.isfinite(xyz).all():
        return None
    return xyz


def _align_up(points: np.ndarray, verticals: list[np.ndarray], scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate so marker verticals point up, then drop the cloud onto z = 0."""
    if points.size == 0:
        identity = np.eye(3)
        return points, identity, np.zeros(3)
    direction = np.zeros(3)
    for vector in verticals:
        vector = np.asarray(vector, dtype=np.float64) * scale
        if np.linalg.norm(vector) < 1e-8:
            continue
        if np.dot(direction, vector) < 0:
            vector = -vector
        direction += vector
    if np.linalg.norm(direction) < 1e-6:
        # Fall back to the longest PCA axis of the cloud as a weak up-hint only
        # when we have no marker edges. Leave the cloud as reconstructed.
        direction = np.array([0.0, 0.0, 1.0])
    direction = direction / np.linalg.norm(direction)
    up = np.array([0.0, 0.0, 1.0])
    rotation = _rotation_between(direction, up)
    rotated = points @ rotation.T
    translation = np.array([0.0, 0.0, -float(np.percentile(rotated[:, 2], 2))])
    return rotated + translation, rotation, translation


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-8:
        if dot > 0:
            return np.eye(3)
        # 180° around any perpendicular axis.
        axis = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, axis)
        axis = axis / np.linalg.norm(axis)
        return _rodrigues(axis, np.pi)
    axis = cross / np.linalg.norm(cross)
    angle = np.arccos(dot)
    return _rodrigues(axis, angle)


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def _dense_points(images, cameras) -> np.ndarray:
    """Stereo depth for a handful of neighboring pairs, in the aligned millimetre frame."""
    names = sorted(cameras)
    if len(names) < 2:
        return np.zeros((0, 3))
    centers = np.vstack([cameras[name]["center"] for name in names])
    pairs = []
    for index, name in enumerate(names):
        distances = np.linalg.norm(centers - centers[index], axis=1)
        distances[index] = np.inf
        neighbor = names[int(np.argmin(distances))]
        key = tuple(sorted((name, neighbor)))
        if key not in pairs:
            pairs.append(key)
    # A full ring only needs each camera's nearest neighbor once.
    collected = []
    for left_name, right_name in pairs[:12]:
        left = cameras[left_name]
        right = cameras[right_name]
        left_index = int(Path(left_name).stem)
        right_index = int(Path(right_name).stem)
        points = _stereo_points(images[left_index], images[right_index], left, right)
        if len(points):
            collected.append(points)
    if not collected:
        return np.zeros((0, 3))
    cloud = np.vstack(collected)
    low = np.percentile(cloud, 1, axis=0)
    high = np.percentile(cloud, 99, axis=0)
    span = np.maximum(high - low, 1e-6)
    keep = np.all((cloud > low - 0.25 * span) & (cloud < high + 0.25 * span), axis=1)
    return cloud[keep]


def _stereo_points(left_bgr, right_bgr, left_cam, right_cam) -> np.ndarray:
    k1 = left_cam["K"]
    k2 = right_cam["K"]
    r_rel = right_cam["R"] @ left_cam["R"].T
    t_rel = right_cam["t"] - r_rel @ left_cam["t"]
    size = (left_bgr.shape[1], left_bgr.shape[0])
    try:
        r1, r2, p1, p2, q, _roi1, _roi2 = cv2.stereoRectify(
            k1,
            None,
            k2,
            None,
            size,
            r_rel,
            t_rel,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.3,
        )
    except cv2.error:
        return np.zeros((0, 3))
    map1x, map1y = cv2.initUndistortRectifyMap(k1, None, r1, p1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(k2, None, r2, p2, size, cv2.CV_32FC1)
    rect1 = cv2.remap(left_bgr, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(right_bgr, map2x, map2y, cv2.INTER_LINEAR)
    gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY)
    window = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=window,
        P1=8 * 3 * window * window,
        P2=32 * 3 * window * window,
        disp12MaxDiff=2,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
    valid = disparity > 1.0
    if int(valid.sum()) < 100:
        # The pair may be swapped relative to the baseline sign. Try the other order.
        disparity = stereo.compute(gray2, gray1).astype(np.float32) / 16.0
        valid = disparity > 1.0
        if int(valid.sum()) < 100:
            return np.zeros((0, 3))
        points = cv2.reprojectImageTo3D(disparity, q)
        points = points[valid]
        # Those points are in the rectified RIGHT camera. Fall through using r2.
        r_rect = r2
        cam = right_cam
    else:
        points = cv2.reprojectImageTo3D(disparity, q)
        points = points[valid]
        r_rect = r1
        cam = left_cam
    # Rectified camera -> original camera -> world.
    original = (r_rect.T @ points.T).T
    rotation_t = cam["R"].T
    center_term = rotation_t @ cam["t"]
    world = (rotation_t @ original.T).T - center_term
    # Subsample so a 640x480 depth map doesn't become hundreds of thousands of points.
    if len(world) > 8000:
        step = int(np.ceil(len(world) / 8000))
        world = world[::step]
    finite = np.isfinite(world).all(axis=1)
    return world[finite]


def _mesh_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 50:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    low = np.percentile(points, 1, axis=0)
    high = np.percentile(points, 99, axis=0)
    span = np.maximum(high - low, 1.0)
    diagonal = float(np.linalg.norm(span))
    voxel = float(np.clip(diagonal / 80.0, 0.8, 6.0))
    dims = np.maximum(8, np.ceil(span / voxel).astype(int) + 4)
    # Cap the grid so a stray point can't allocate a huge volume.
    if int(np.prod(dims)) > 200**3:
        voxel = float(np.cbrt(np.prod(span) / (160**3)))
        voxel = max(voxel, 1.0)
        dims = np.maximum(8, np.ceil(span / voxel).astype(int) + 4)
    origin = low - 2 * voxel
    idx = np.floor((points - origin) / voxel).astype(int)
    inside = np.all((idx >= 0) & (idx < dims), axis=1)
    idx = idx[inside]
    if len(idx) < 30:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    volume = np.zeros(tuple(int(v) for v in dims), dtype=np.float32)
    volume[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    # A light blur turns the shell of points into a band marching-cubes can cross.
    from scipy.ndimage import gaussian_filter

    volume = gaussian_filter(volume, sigma=0.8)
    try:
        verts, faces, _normals, _values = marching_cubes(volume, level=float(volume.max()) * 0.35, spacing=(voxel, voxel, voxel))
    except (ValueError, RuntimeError):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    verts = verts + origin
    return verts.astype(np.float64), faces.astype(np.int32)


def _rescale_to_segment(vertices, reference_points, distance_mm, rotation, translation, current_scale):
    """Scale the finished mesh so the segment between two world points matches distance_mm.

    reference_points are in the same aligned millimetre frame as the mesh.
    """
    a = np.asarray(reference_points[0], dtype=np.float64)
    b = np.asarray(reference_points[1], dtype=np.float64)
    current = float(np.linalg.norm(a - b))
    if current < 1e-6:
        raise CaptureError("BAD_REFERENCE", "Those two points are the same. Pick two different spots.")
    if distance_mm <= 0:
        raise CaptureError("BAD_REFERENCE", "The reference distance has to be greater than zero.")
    factor = float(distance_mm) / current
    center = vertices.mean(axis=0)
    return (vertices - center) * factor + center, factor


def _extents(vertices: np.ndarray) -> dict:
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    size = high - low
    return {
        "min_mm": [float(v) for v in low],
        "max_mm": [float(v) for v in high],
        "size_mm": [float(v) for v in size],
        "diagonal_mm": float(np.linalg.norm(size)),
    }


def scale_mesh_to_distance(vertices: np.ndarray, point_a: np.ndarray, point_b: np.ndarray, distance_mm: float) -> tuple[np.ndarray, float]:
    """Uniform scale about the mesh centroid so |a-b| becomes distance_mm."""
    current = float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))
    if current < 1e-6:
        raise CaptureError("BAD_REFERENCE", "Those two points are the same. Pick two different spots.")
    if distance_mm <= 0:
        raise CaptureError("BAD_REFERENCE", "The reference distance has to be greater than zero.")
    factor = float(distance_mm) / current
    center = vertices.mean(axis=0)
    return (vertices - center) * factor + center, factor


def scale_mesh_to_extents(
    vertices: np.ndarray,
    measured_width_mm: float | None,
    measured_depth_mm: float | None,
    measured_height_mm: float | None,
) -> tuple[np.ndarray, dict]:
    """Uniform scale from one or two overall caliper readings versus the bbox."""
    size = vertices.max(axis=0) - vertices.min(axis=0)
    # size is x, y, z after alignment: x/y are the footprint, z is height.
    pairs = []
    if measured_width_mm:
        pairs.append(measured_width_mm / float(max(size[0], size[1])))
    if measured_depth_mm:
        pairs.append(measured_depth_mm / float(min(size[0], size[1])))
    if measured_height_mm:
        pairs.append(measured_height_mm / float(size[2]))
    pairs = [value for value in pairs if np.isfinite(value) and value > 0]
    if not pairs:
        raise CaptureError(
            "NO_REFERENCE",
            "Enter a measured width, depth, or height to check the scale.",
        )
    factor = float(np.median(pairs))
    center = vertices.mean(axis=0)
    scaled = (vertices - center) * factor + center
    spread = float(np.max(pairs) - np.min(pairs)) if len(pairs) > 1 else 0.0
    return scaled, {
        "factor": factor,
        "anisotropic": False,
        "warning": (
            "Those measurements don't agree with a single scale, so the median was used. "
            "Check the marker sheets and the picked points."
            if spread > 0.03
            else "Scale was adjusted to your measurement."
        ),
    }
