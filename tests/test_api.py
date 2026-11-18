"""The shop API: health, a printable sheet, and a flat job through to DXF."""

import io
import time
import zipfile

import cv2
import numpy as np
from fastapi.testclient import TestClient

from partcapture.main import app
from partcapture.synthetic import rectangle, render_photo, render_plane, work_area
from partcapture.pipeline.mosaic import Pose2D
from partcapture.pipeline.sheets import make_instance


def test_health_and_marker_pdf():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True
        page = client.get("/")
        assert page.status_code == 200
        assert b"TrueShape" in page.content
        pdf = client.get("/api/sheets/pdf?layout=letter&count=2&start=1")
        assert pdf.status_code == 200
        assert pdf.content.startswith(b"%PDF")
        assert "attachment" in pdf.headers["content-disposition"]


def test_flat_sample_job_reaches_a_dxf():
    with TestClient(app) as client:
        created = client.post("/api/demo/flat")
        assert created.status_code == 200
        job_id = created.json()["id"]
        job = None
        for _ in range(50):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"done", "error"}:
                break
            time.sleep(0.2)
        assert job["status"] == "done", job
        length = job["result"]["measurements"]["length_mm"]
        width = job["result"]["measurements"]["width_mm"]
        assert abs(length - 100) < 0.5, length
        assert abs(width - 60) < 0.5, width
        assert job["result"]["accuracy"]["estimated_error_mm"] < 1.0
        dxf = client.get(f"/api/jobs/{job_id}/result.dxf?unit=in")
        assert dxf.status_code == 200
        assert b"SECTION" in dxf.content
        listed = client.get("/api/jobs").json()["jobs"]
        assert any(item["id"] == job_id for item in listed)


def _jpeg_of_plate() -> bytes:
    left, bottom, right, top = work_area("letter", 0)
    cx, cy = (left + right) / 2, (bottom + top) / 2
    outer = rectangle(cx - 40, cy - 20, cx + 40, cy + 20)
    plane = render_plane([(make_instance("letter", 0), Pose2D())], [{"outer": outer, "color": (40, 70, 140)}], px_per_mm=6)
    photo = render_photo(plane, np.array([cx, cy, 500.0]), np.array([cx, cy, 0.0]), image_size=(900, 700), focal_scale=0.9)
    ok, encoded = cv2.imencode(".jpg", photo, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    return encoded.tobytes()


def test_batch_accepts_a_zip_and_ignores_junk():
    jpeg = _jpeg_of_plate()
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as archive:
        archive.writestr("parts/remote.jpg", jpeg)
        archive.writestr("notes.txt", b"not a photo")
        archive.writestr("__MACOSX/._remote.jpg", b"junk")
    blob.seek(0)
    with TestClient(app) as client:
        created = client.post(
            "/api/jobs/batch",
            files={"files": ("shots.zip", blob.getvalue(), "application/zip")},
            data={"thickness_mm": "0", "profile": "prism", "units": "mm"},
        )
        assert created.status_code == 200, created.text
        assert created.json()["count"] == 1
        job_id = created.json()["id"]
        job = None
        for _ in range(80):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"done", "error"}:
                break
            time.sleep(0.25)
        assert job["status"] == "done", job
        photo = job["result"]["photos"][0]
        assert photo["ok"], photo
        assert abs(photo["length_mm"] - 80) < 2.0
        assert abs(photo["width_mm"] - 40) < 2.0
        dxf = client.get(f"/api/jobs/{job_id}/items/0/part.dxf")
        assert dxf.status_code == 200
        assert b"LWPOLYLINE" in dxf.content or b"SECTION" in dxf.content
        packed = client.get(f"/api/jobs/{job_id}/batch.zip")
        assert packed.status_code == 200
        with zipfile.ZipFile(io.BytesIO(packed.content)) as archive:
            assert any(name.endswith(".dxf") for name in archive.namelist())
