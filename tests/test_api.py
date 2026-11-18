"""The shop API: health, a printable sheet, and a flat job through to DXF."""

import time

from fastapi.testclient import TestClient

from partcapture.main import app


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
