"""Marker-sheet PDFs are true scale and carry unique IDs per sheet."""

import io

from pypdf import PdfReader

from partcapture.pipeline.pdf_sheets import build_marker_pdf
from partcapture.pipeline.sheets import LAYOUTS, make_instance


def _page_mm(page) -> tuple[float, float]:
    box = page.mediabox
    return float(box.width) / 72 * 25.4, float(box.height) / 72 * 25.4


def test_letter_pdf_is_true_size_and_warns_about_scaling():
    data = build_marker_pdf("letter", sheet_count=1)
    reader = PdfReader(io.BytesIO(data))
    assert len(reader.pages) == 1
    width, height = _page_mm(reader.pages[0])
    layout = LAYOUTS["letter"]
    assert abs(width - layout.page_width_mm) < 0.2
    assert abs(height - layout.page_height_mm) < 0.2
    text = reader.pages[0].extract_text()
    assert "100%" in text
    assert "100 mm" in text
    assert "Sheet 1" in text


def test_numbered_set_has_unique_marker_ids():
    data = build_marker_pdf("letter", sheet_count=3)
    reader = PdfReader(io.BytesIO(data))
    assert len(reader.pages) == 3
    ids = []
    for index in range(3):
        inst = make_instance("letter", index)
        ids.extend(inst.markers_by_id)
        text = reader.pages[index].extract_text()
        assert f"Sheet {index + 1}" in text
    assert len(ids) == len(set(ids))


def test_large_board_is_24_by_36_inches():
    data = build_marker_pdf("large", sheet_count=1)
    reader = PdfReader(io.BytesIO(data))
    width, height = _page_mm(reader.pages[0])
    assert abs(width - 24 * 25.4) < 0.3
    assert abs(height - 36 * 25.4) < 0.3
    text = reader.pages[0].extract_text()
    assert "400 mm" in text
    assert "100%" in text
