"""Turn uploaded photos into files on disk and a job record."""

from __future__ import annotations

import io
import json
import shutil
import threading
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None

from partcapture.config import FLAT_MAX_EDGE, JOBS_DIR, OBJECT_MAX_EDGE, ensure_dirs
from partcapture.db import get_db
from partcapture.pipeline.dxf_export import write_dxf
from partcapture.pipeline.errors import CaptureError
from partcapture.pipeline.geometry import outline_from_json
from partcapture.pipeline.mesh_export import write_obj, write_stl
from partcapture.pipeline.photogrammetry import reconstruct_object, scale_mesh_to_distance, scale_mesh_to_extents
from partcapture.pipeline.photo_measure import measure_photo
from partcapture.pipeline.twod import measure_flat_images

_worker: "Worker | None" = None


def job_dir(job_id: str) -> Path:
    ensure_dirs()
    return JOBS_DIR / job_id


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def focal_35mm_from_bytes(data: bytes) -> float | None:
    """35 mm-equivalent focal length from EXIF, before the pixels are re-saved."""
    try:
        image = Image.open(io.BytesIO(data))
        exif = image.getexif()
        sub = exif.get_ifd(0x8769)
        value = sub.get(0xA405)
        if value:
            return float(value)
    except Exception:
        return None
    return None


def decode_image(data: bytes) -> np.ndarray:
    image, _focal = decode_image_with_focal(data)
    return image


def decode_image_with_focal(data: bytes) -> tuple[np.ndarray, float | None]:
    focal = focal_35mm_from_bytes(data)
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    array = np.array(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR), focal


def images_from_zip(data: bytes) -> list[tuple[str, bytes]]:
    """Pull photos out of a zip. Skip folders, Apple junk, and non-images."""
    found = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("That zip file could not be opened.") from exc
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = Path(info.filename).name
        if not name or name.startswith(".") or name.startswith("__MACOSX") or info.filename.startswith("__MACOSX"):
            continue
        if Path(name).suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if info.file_size > 40 * 1024 * 1024:
            continue
        found.append((name, archive.read(info)))
        if len(found) >= 40:
            break
    return found


def downscale(image: np.ndarray, max_edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_edge:
        return image
    scale = max_edge / long_edge
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def save_inputs(job_id: str, images: list[np.ndarray]) -> None:
    folder = job_dir(job_id) / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        cv2.imwrite(str(folder / f"{index:03d}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def load_inputs(job_id: str) -> list[np.ndarray]:
    folder = job_dir(job_id) / "inputs"
    paths = sorted(folder.glob("*.jpg"))
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
    return images


def _jsonable(result: dict) -> dict:
    skip = {"mosaic", "mask", "valid", "preview", "fitted", "vertices", "faces"}
    out = {}
    for key, value in result.items():
        if key in skip:
            continue
        out[key] = _convert(value)
    return out


def _convert(value):
    if isinstance(value, dict):
        return {k: _convert(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def process_flat(job_id: str, measured_length_mm: float | None = None, measured_width_mm: float | None = None, mask: np.ndarray | None = None) -> None:
    db = get_db()
    images = load_inputs(job_id)
    db.update(job_id, progress=0.2, message="Looking for markers")
    result = measure_flat_images(
        images,
        user_mask=mask,
        measured_length_mm=measured_length_mm,
        measured_width_mm=measured_width_mm,
    )
    folder = job_dir(job_id)
    cv2.imwrite(str(folder / "preview.jpg"), result["preview"], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    cv2.imwrite(str(folder / "mosaic.jpg"), result["mosaic"], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    cv2.imwrite(str(folder / "mask.png"), result["mask"])
    note = (
        f"TrueShape  estimated accuracy +/- {result['accuracy']['estimated_error_mm']:.2f} mm. "
        "Verify critical dimensions with calipers."
    )
    write_dxf(result["fitted"], folder / "result_mm.dxf", unit="mm", note=note)
    payload = _jsonable(result)
    db.update(job_id, status="done", progress=1.0, message="Ready", result_json=json.dumps(payload))


def process_object(job_id: str) -> None:
    db = get_db()
    images = load_inputs(job_id)

    def progress(fraction: float, message: str) -> None:
        db.update(job_id, progress=fraction, message=message)

    result = reconstruct_object(images, job_dir(job_id) / "work", progress=progress)
    folder = job_dir(job_id)
    write_stl(folder / "model.stl", result["vertices"], result["faces"])
    write_obj(folder / "model.obj", result["vertices"], result["faces"])
    np.savez_compressed(folder / "mesh.npz", vertices=result["vertices"], faces=result["faces"])
    payload = _jsonable(result)
    db.update(job_id, status="done", progress=1.0, message="Ready", result_json=json.dumps(payload))


def rescale_object(
    job_id: str,
    *,
    point_a: list[float] | None = None,
    point_b: list[float] | None = None,
    distance_mm: float | None = None,
    measured_width_mm: float | None = None,
    measured_depth_mm: float | None = None,
    measured_height_mm: float | None = None,
) -> dict:
    folder = job_dir(job_id)
    mesh = np.load(folder / "mesh.npz")
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    record = get_db().get(job_id)
    payload = json.loads(record["result_json"] or "{}")
    if point_a and point_b and distance_mm:
        vertices, factor = scale_mesh_to_distance(
            vertices, np.array(point_a), np.array(point_b), float(distance_mm)
        )
        note = f"Scale set from a {distance_mm:.2f} mm reference (factor {factor:.4f})."
    else:
        vertices, info = scale_mesh_to_extents(
            vertices, measured_width_mm, measured_depth_mm, measured_height_mm
        )
        note = info["warning"]
        factor = info["factor"]
    from partcapture.pipeline.photogrammetry import _extents
    from partcapture.pipeline.accuracy import estimate_3d

    extents = _extents(vertices)
    accuracy = estimate_3d(extents["diagonal_mm"], 0.0)
    accuracy["anchored_to_calipers"] = True
    write_stl(folder / "model.stl", vertices, faces)
    write_obj(folder / "model.obj", vertices, faces)
    np.savez_compressed(folder / "mesh.npz", vertices=vertices, faces=faces)
    payload["extents"] = extents
    payload["accuracy"] = accuracy
    payload["scaled_by"] = "reference"
    payload.setdefault("warnings", []).append(note)
    payload["scale_factor_applied"] = factor
    get_db().update(job_id, result_json=json.dumps(_convert(payload)), message="Scale updated")
    return payload


def export_dxf(job_id: str, unit: str) -> Path:
    folder = job_dir(job_id)
    if unit == "mm" and (folder / "result_mm.dxf").exists():
        return folder / "result_mm.dxf"
    record = get_db().get(job_id)
    payload = json.loads(record["result_json"])
    fitted = outline_from_json(payload["geometry"])
    accuracy = payload.get("accuracy", {}).get("estimated_error_mm")
    note = None
    if accuracy is not None:
        note = f"TrueShape estimated accuracy +/- {accuracy:.2f} mm. Verify with calipers."
    target = folder / f"result_{unit}.dxf"
    write_dxf(fitted, target, unit=unit, note=note)
    return target


def save_batch_inputs(job_id: str, items: list[tuple[str, np.ndarray, float | None]]) -> None:
    folder = job_dir(job_id) / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    meta = []
    for index, (name, image, focal) in enumerate(items):
        filename = f"{index:03d}.jpg"
        cv2.imwrite(str(folder / filename), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        meta.append({"file": filename, "name": name, "focal_35mm": focal})
    (folder / "batch.json").write_text(json.dumps(meta), encoding="utf-8")


def process_batch(job_id: str) -> None:
    db = get_db()
    folder = job_dir(job_id)
    meta = json.loads((folder / "inputs" / "batch.json").read_text(encoding="utf-8"))
    options = json.loads((folder / "options.json").read_text(encoding="utf-8"))
    items_dir = folder / "items"
    items_dir.mkdir(exist_ok=True)
    photos = []
    dxf_paths = []
    for index, item in enumerate(meta):
        db.update(job_id, progress=(index + 0.2) / max(len(meta), 1), message=f"Measuring {item['name']}")
        image = cv2.imread(str(folder / "inputs" / item["file"]), cv2.IMREAD_COLOR)
        result = measure_photo(
            image,
            thickness_mm=options.get("thickness_mm"),
            profile=options.get("profile") or "prism",
            caliper_mm=options.get("caliper_mm"),
            units=options.get("units") or "mm",
            bg=options.get("bg") or "auto",
            focal_35mm=item.get("focal_35mm"),
        )
        entry = result.to_json()
        entry["name"] = item["name"]
        entry["index"] = index
        if result.overlay_bgr is not None:
            cv2.imwrite(str(items_dir / f"{index:03d}.jpg"), result.overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            entry["overlay"] = True
        if result.dxf_bytes:
            dxf_path = items_dir / f"{index:03d}.dxf"
            dxf_path.write_bytes(result.dxf_bytes)
            dxf_paths.append((item["name"], dxf_path))
            entry["dxf"] = True
        photos.append(entry)
    if dxf_paths:
        used = set()
        with zipfile.ZipFile(folder / "batch.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for index, (name, path) in enumerate(dxf_paths):
                stem = Path(name).stem or "part"
                filename = f"{stem}.dxf"
                if filename in used:
                    filename = f"{index:02d}-{stem}.dxf"
                used.add(filename)
                archive.write(path, filename)
    ok = sum(1 for photo in photos if photo["ok"])
    db.update(
        job_id,
        status="done",
        progress=1.0,
        message=f"{ok} of {len(photos)} photos measured",
        result_json=json.dumps({"photos": photos, "options": options}),
    )


def run_job(job_id: str) -> None:
    db = get_db()
    record = db.get(job_id)
    if record is None:
        return
    try:
        if record["mode"] == "flat":
            process_flat(job_id)
        elif record["mode"] == "object":
            process_object(job_id)
        elif record["mode"] == "batch":
            process_batch(job_id)
        else:
            raise CaptureError("BAD_JOB", "Unknown job type.")
    except CaptureError as exc:
        db.update(job_id, status="error", progress=1.0, message=exc.message, error=exc.code)
    except Exception as exc:  # noqa: BLE001 — last-resort log for the shop UI
        db.update(
            job_id,
            status="error",
            progress=1.0,
            message="Something went wrong while measuring. The photos may be unusable — retake them and try again.",
            error="INTERNAL",
        )
        # Keep the technical reason in the job folder for whoever is deploying this.
        (job_dir(job_id)).mkdir(parents=True, exist_ok=True)
        (job_dir(job_id) / "error.txt").write_text(repr(exc), encoding="utf-8")


class Worker:
    def __init__(self):
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="trueshape-jobs", daemon=True)
        self._thread.start()

    def kick(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _loop(self) -> None:
        db = get_db()
        while not self._stop.is_set():
            job_id = db.claim_next()
            if job_id is None:
                self._wake.wait(0.4)
                self._wake.clear()
                continue
            run_job(job_id)


def get_worker() -> Worker:
    global _worker
    if _worker is None:
        _worker = Worker()
    return _worker


def delete_job(job_id: str) -> None:
    get_db().delete(job_id)
    folder = job_dir(job_id)
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
