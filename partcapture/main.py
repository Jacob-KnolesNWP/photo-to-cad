"""TrueShape HTTP API and the mobile page."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from partcapture import __version__
from partcapture.config import (
    FLAT_MAX_EDGE,
    MAX_FLAT_PHOTOS,
    MAX_OBJECT_PHOTOS,
    MAX_UPLOAD_BYTES,
    MIN_OBJECT_PHOTOS,
    OBJECT_MAX_EDGE,
    STATIC_DIR,
    ensure_dirs,
)
from partcapture.db import get_db
from partcapture.jobs import (
    decode_image,
    decode_image_with_focal,
    delete_job,
    downscale,
    export_dxf,
    get_worker,
    images_from_zip,
    job_dir,
    process_flat,
    rescale_object,
    save_batch_inputs,
    save_inputs,
)
from partcapture.pipeline.errors import CaptureError
from partcapture.pipeline.pdf_sheets import build_marker_pdf
from partcapture.pipeline.sheets import layout_catalog, max_sheets
from partcapture.synthetic import sample_flat_plate
from partcapture.synthetic3d import sample_object_views


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_dirs()
    worker = get_worker()
    worker.start()
    yield
    worker.stop()


app = FastAPI(title="TrueShape", version=__version__, lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"ok": True, "name": "TrueShape", "version": __version__}


@app.get("/api/sheets")
def sheets():
    return {"layouts": layout_catalog()}


@app.get("/api/sheets/pdf")
def sheet_pdf(layout: str = "letter", count: int = 1, start: int = 1):
    if layout not in {"letter", "tabloid", "large"}:
        raise HTTPException(status_code=400, detail="Choose letter, tabloid, or large.")
    if count < 1 or count > max_sheets(layout):
        raise HTTPException(status_code=400, detail=f"This layout prints 1–{max_sheets(layout)} sheets at a time.")
    if start < 1:
        raise HTTPException(status_code=400, detail="Sheet numbers start at 1.")
    try:
        pdf = build_marker_pdf(layout, sheet_count=count, start_index=start - 1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    filename = f"trueshape-{layout}-sheet{start}.pdf" if count == 1 else f"trueshape-{layout}-{count}-sheets.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def _read_uploads(files: list[UploadFile], max_edge: int, limit: int) -> list[np.ndarray]:
    if not files:
        raise HTTPException(status_code=400, detail="Add at least one photo.")
    if len(files) > limit:
        raise HTTPException(status_code=400, detail=f"That's {len(files)} photos. The limit is {limit}.")
    images = []
    for index, upload in enumerate(files):
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail=f"Photo {index + 1} is over 25 MB.")
        if not data:
            raise HTTPException(status_code=400, detail=f"Photo {index + 1} was empty.")
        try:
            image = decode_image(data)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Couldn't read photo {index + 1}. Use a JPEG or PNG.") from exc
        images.append(downscale(image, max_edge))
    return images


def _enqueue(mode: str, images: list[np.ndarray]) -> dict:
    job_id = uuid4().hex
    save_inputs(job_id, images)
    get_db().create(job_id, mode)
    get_worker().kick()
    return {"id": job_id, "mode": mode, "status": "queued"}


@app.post("/api/jobs/flat")
async def create_flat_job(files: list[UploadFile] = File(...)):
    images = await _read_uploads(files, FLAT_MAX_EDGE, MAX_FLAT_PHOTOS)
    return _enqueue("flat", images)


@app.post("/api/jobs/object")
async def create_object_job(files: list[UploadFile] = File(...)):
    if len(files) < MIN_OBJECT_PHOTOS:
        raise HTTPException(
            status_code=400,
            detail=f"3D capture needs at least {MIN_OBJECT_PHOTOS} photos. You added {len(files)}.",
        )
    images = await _read_uploads(files, OBJECT_MAX_EDGE, MAX_OBJECT_PHOTOS)
    return _enqueue("object", images)


@app.post("/api/jobs/batch")
async def create_batch_job(
    files: list[UploadFile] = File(...),
    thickness_mm: float | None = Form(None),
    profile: str = Form("prism"),
    caliper_mm: float | None = Form(None),
    units: str = Form("mm"),
    bg: str = Form("auto"),
):
    """One job, one part setup, many photos (loose files and/or zips)."""
    if profile not in {"prism", "rounded"}:
        raise HTTPException(status_code=400, detail="Edge profile must be prism (square) or rounded.")
    if units not in {"mm", "in"}:
        raise HTTPException(status_code=400, detail="Units must be mm or in.")
    collected: list[tuple[str, np.ndarray, float | None]] = []
    for upload in files:
        data = await upload.read()
        if not data:
            continue
        name = upload.filename or "photo"
        blobs: list[tuple[str, bytes]] = []
        if name.lower().endswith(".zip") or (upload.content_type or "").endswith("zip"):
            try:
                blobs = images_from_zip(data)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            blobs = [(name, data)]
        for item_name, blob in blobs:
            if len(blob) > MAX_UPLOAD_BYTES:
                continue
            try:
                image, focal = decode_image_with_focal(blob)
            except Exception:
                continue
            collected.append((item_name, downscale(image, FLAT_MAX_EDGE), focal))
            if len(collected) >= 40:
                break
        if len(collected) >= 40:
            break
    if not collected:
        raise HTTPException(
            status_code=400,
            detail="No photos found. Add JPEG, PNG, or HEIC pictures, or a zip of them.",
        )
    job_id = uuid4().hex
    save_batch_inputs(job_id, collected)
    (job_dir(job_id) / "options.json").write_text(
        json.dumps(
            {
                "thickness_mm": thickness_mm,
                "profile": profile,
                "caliper_mm": caliper_mm,
                "units": units,
                "bg": bg,
            }
        ),
        encoding="utf-8",
    )
    get_db().create(job_id, "batch")
    get_worker().kick()
    return {"id": job_id, "mode": "batch", "status": "queued", "count": len(collected)}


@app.get("/api/jobs/{job_id}/batch.zip")
def batch_zip(job_id: str):
    return _file(job_id, "batch.zip", "application/zip", f"trueshape-{job_id[:8]}.zip")


@app.get("/api/jobs/{job_id}/items/{index}/overlay.jpg")
def item_overlay(job_id: str, index: int):
    return _file(job_id, f"items/{index:03d}.jpg", "image/jpeg")


@app.get("/api/jobs/{job_id}/items/{index}/part.dxf")
def item_dxf(job_id: str, index: int):
    return _file(job_id, f"items/{index:03d}.dxf", "application/dxf", f"part-{index + 1}.dxf")


@app.post("/api/demo/flat")
def demo_flat():
    images, truth = sample_flat_plate()
    payload = _enqueue("flat", images)
    payload["sample"] = truth
    return payload


@app.post("/api/demo/object")
def demo_object():
    images, truth = sample_object_views()
    payload = _enqueue("object", images)
    payload["sample"] = truth
    return payload


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": get_db().list_recent()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    record = get_db().get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="That job is gone.")
    result = json.loads(record["result_json"]) if record.get("result_json") else None
    return {
        "id": record["id"],
        "mode": record["mode"],
        "status": record["status"],
        "progress": record["progress"],
        "message": record["message"],
        "error": record["error"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "result": result,
    }


@app.post("/api/jobs/{job_id}/flat-update")
async def update_flat_job(
    job_id: str,
    measured_length_mm: float | None = Form(None),
    measured_width_mm: float | None = Form(None),
    mask: UploadFile | None = File(None),
):
    record = get_db().get(job_id)
    if record is None or record["mode"] != "flat":
        raise HTTPException(status_code=404, detail="No flat job with that id.")
    if record["status"] != "done":
        raise HTTPException(status_code=409, detail="Wait until the measurement finishes.")
    mask_array = None
    if mask is not None and mask.filename:
        data = await mask.read()
        if data:
            decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if decoded is None:
                raise HTTPException(status_code=400, detail="Couldn't read the touch-up mask.")
            mask_array = decoded
    try:
        process_flat(job_id, measured_length_mm=measured_length_mm, measured_width_mm=measured_width_mm, mask=mask_array)
    except CaptureError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    return await _job_body(job_id)


async def _job_body(job_id: str):
    return get_job(job_id)


@app.post("/api/jobs/{job_id}/object-scale")
async def update_object_scale(
    job_id: str,
    distance_mm: float | None = Form(None),
    ax: float | None = Form(None),
    ay: float | None = Form(None),
    az: float | None = Form(None),
    bx: float | None = Form(None),
    by: float | None = Form(None),
    bz: float | None = Form(None),
    measured_width_mm: float | None = Form(None),
    measured_depth_mm: float | None = Form(None),
    measured_height_mm: float | None = Form(None),
):
    record = get_db().get(job_id)
    if record is None or record["mode"] != "object":
        raise HTTPException(status_code=404, detail="No 3D job with that id.")
    if record["status"] != "done":
        raise HTTPException(status_code=409, detail="Wait until the model finishes.")
    point_a = point_b = None
    if None not in (ax, ay, az, bx, by, bz):
        point_a = [ax, ay, az]
        point_b = [bx, by, bz]
    try:
        rescale_object(
            job_id,
            point_a=point_a,
            point_b=point_b,
            distance_mm=distance_mm,
            measured_width_mm=measured_width_mm,
            measured_depth_mm=measured_depth_mm,
            measured_height_mm=measured_height_mm,
        )
    except CaptureError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    return get_job(job_id)


@app.delete("/api/jobs/{job_id}")
def remove_job(job_id: str):
    if get_db().get(job_id) is None:
        raise HTTPException(status_code=404, detail="That job is gone.")
    delete_job(job_id)
    return {"ok": True}


def _file(job_id: str, name: str, media: str, download: str | None = None):
    path = job_dir(job_id) / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="That file isn't ready yet.")
    return FileResponse(path, media_type=media, filename=download)


@app.get("/api/jobs/{job_id}/preview.jpg")
def preview(job_id: str):
    return _file(job_id, "preview.jpg", "image/jpeg")


@app.get("/api/jobs/{job_id}/mosaic.jpg")
def mosaic(job_id: str):
    return _file(job_id, "mosaic.jpg", "image/jpeg")


@app.get("/api/jobs/{job_id}/mask.png")
def mask(job_id: str):
    return _file(job_id, "mask.png", "image/png")


@app.get("/api/jobs/{job_id}/result.dxf")
def dxf(job_id: str, unit: str = "mm"):
    if unit not in {"mm", "in"}:
        raise HTTPException(status_code=400, detail="Unit must be mm or in.")
    record = get_db().get(job_id)
    if record is None or record["status"] != "done":
        raise HTTPException(status_code=404, detail="The DXF isn't ready yet.")
    path = export_dxf(job_id, unit)
    filename = f"trueshape-{job_id[:8]}.{unit}.dxf"
    return FileResponse(path, media_type="application/dxf", filename=filename)


@app.get("/api/jobs/{job_id}/model.stl")
def stl(job_id: str):
    return _file(job_id, "model.stl", "model/stl", f"trueshape-{job_id[:8]}.stl")


@app.get("/api/jobs/{job_id}/model.obj")
def obj(job_id: str):
    return _file(job_id, "model.obj", "model/obj", f"trueshape-{job_id[:8]}.obj")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
