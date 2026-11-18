"""Printable marker-sheet PDFs at true scale.

The scale bar is drawn as vectors in PDF millimetres, not as a raster image,
so a ruler check is about the printer, not about our pixel grid.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from partcapture.pipeline.sheets import SheetLayout, get_layout, make_instance, max_sheets


def render_marker_image(marker_id: int, side_px: int) -> np.ndarray:
    from partcapture.pipeline.sheets import get_dictionary

    image = cv2.aruco.generateImageMarker(get_dictionary(), int(marker_id), int(side_px))
    return image


def build_marker_pdf(layout_key: str, sheet_count: int = 1, start_index: int = 0) -> bytes:
    layout = get_layout(layout_key)
    limit = max_sheets(layout_key)
    if sheet_count < 1:
        raise ValueError("Print at least one sheet.")
    if start_index < 0 or start_index + sheet_count > limit:
        raise ValueError(
            f"This {layout.label} set can include sheets 1–{limit} in one download "
            f"(so every marker ID stays unique). You asked for {sheet_count} starting at {start_index + 1}."
        )
    buffer = io.BytesIO()
    pdf = canvas.Canvas(
        buffer,
        pagesize=(layout.page_width_mm * mm, layout.page_height_mm * mm),
    )
    pdf.setTitle(f"TrueShape markers — {layout.label}")
    pdf.setAuthor("TrueShape / Northland Wood Products")
    for offset in range(sheet_count):
        instance = make_instance(layout_key, start_index + offset)
        _draw_page(pdf, instance, page_number=offset + 1, page_count=sheet_count)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _draw_page(pdf: canvas.Canvas, instance, page_number: int, page_count: int) -> None:
    layout: SheetLayout = instance.layout
    # Quiet zone is the paper itself. Markers are placed with a white margin.
    side_px = max(80, int(round(layout.marker_size_mm * 12)))  # ~300 dpi
    for marker_id, spec in instance.markers_by_id.items():
        image = render_marker_image(marker_id, side_px)
        pil = Image.fromarray(image)
        encoded = io.BytesIO()
        pil.save(encoded, format="PNG")
        encoded.seek(0)
        x = (spec.x_mm - spec.size_mm / 2.0) * mm
        y = (spec.y_mm - spec.size_mm / 2.0) * mm
        pdf.drawImage(
            ImageReader(encoded),
            x,
            y,
            width=spec.size_mm * mm,
            height=spec.size_mm * mm,
            mask="auto",
        )
    _draw_scale_bar(pdf, layout)
    _draw_footer(pdf, instance, page_number, page_count)
    _draw_crop_marks(pdf, layout)


def _draw_scale_bar(pdf: canvas.Canvas, layout: SheetLayout) -> None:
    x0 = layout.scale_bar_x_mm * mm
    x1 = (layout.scale_bar_x_mm + layout.scale_bar_mm) * mm
    y = layout.scale_bar_y_mm * mm
    thick = layout.scale_bar_thickness_mm * mm
    pdf.setFillColorRGB(0, 0, 0)
    pdf.rect(x0, y - thick / 2.0, x1 - x0, thick, stroke=0, fill=1)
    # End ticks, longer than the bar is thick, so a ruler can sit on them.
    tick = 4.5 * mm
    pdf.setStrokeColorRGB(0, 0, 0)
    pdf.setLineWidth(0.6)
    pdf.line(x0, y - tick, x0, y + tick)
    pdf.line(x1, y - tick, x1, y + tick)
    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont("Helvetica-Bold", 8)
    label = f"{layout.scale_bar_mm:.0f} mm"
    pdf.drawString(x0, y + tick + 1.5 * mm, label)


def _draw_footer(pdf: canvas.Canvas, instance, page_number: int, page_count: int) -> None:
    layout = instance.layout
    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont("Helvetica-Bold", 8)
    y = 3.2 * mm
    left = 8 * mm
    pdf.drawString(
        left,
        y + 8.2 * mm,
        "PRINT AT 100%  —  do not use Fit to Page, Shrink, or a poster tile.",
    )
    pdf.setFont("Helvetica", 7.5)
    pdf.drawString(
        left,
        y + 4.6 * mm,
        f"The black bar must measure {layout.scale_bar_mm:.0f} mm with a ruler. "
        "If it does not, reprint before measuring a part.",
    )
    ids = sorted(instance.markers_by_id)
    pdf.drawString(
        left,
        y + 1.2 * mm,
        f"TrueShape  ·  {layout.label}  ·  Sheet {instance.sheet_index + 1} "
        f"({page_number} of {page_count})  ·  marker IDs {ids[0]}–{ids[-1]}",
    )


def _draw_crop_marks(pdf: canvas.Canvas, layout: SheetLayout) -> None:
    pdf.setStrokeColorRGB(0, 0, 0)
    pdf.setLineWidth(0.4)
    inset = 3.5 * mm
    mark = 6 * mm
    w = layout.page_width_mm * mm
    h = layout.page_height_mm * mm
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    for x, y in corners:
        sx = 1 if x == 0 else -1
        sy = 1 if y == 0 else -1
        pdf.line(x + sx * inset, y, x + sx * (inset + mark), y)
        pdf.line(x, y + sy * inset, x, y + sy * (inset + mark))
