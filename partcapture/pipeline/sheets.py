"""Printable marker-sheet geometry.

Each physical sheet carries a unique block of ArUco IDs from DICT_5X5_1000.
Positions are in millimetres with the origin at the bottom-left of the page
and +Y toward the top of the page, matching CAD and the PDF coordinate system.

A sheet's own coordinates are local. When several sheets are laid around a
large part, their poses in a shared plane are solved from photos that see
more than one sheet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DICTIONARY_ID = cv2.aruco.DICT_5X5_1000
# One dictionary, split into layout blocks so a letter sheet never collides
# with a tabloid or a print-shop board.
LAYOUT_ID_BASE = {
    "letter": 0,
    "tabloid": 500,
    "large": 750,
}
LAYOUT_STRIDE = {
    "letter": 40,
    "tabloid": 50,
    "large": 60,
}
LAYOUT_MAX_SHEETS = {
    "letter": 12,  # IDs 0..479
    "tabloid": 5,  # IDs 500..749
    "large": 4,  # IDs 750..989
}


@dataclass(frozen=True)
class MarkerSpec:
    local_index: int
    x_mm: float  # center
    y_mm: float
    size_mm: float

    def corners_local(self) -> np.ndarray:
        """Canonical ArUco corner order: TL, TR, BR, BL of the printed marker.

        TL is the marker pattern's top-left, which sits toward +Y on the page
        when the sheet is printed upright.
        """
        h = self.size_mm / 2.0
        return np.array(
            [
                [self.x_mm - h, self.y_mm + h],
                [self.x_mm + h, self.y_mm + h],
                [self.x_mm + h, self.y_mm - h],
                [self.x_mm - h, self.y_mm - h],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class SheetLayout:
    key: str
    label: str
    page_width_mm: float
    page_height_mm: float
    marker_size_mm: float
    footer_height_mm: float
    scale_bar_mm: float
    scale_bar_x_mm: float
    scale_bar_y_mm: float
    scale_bar_thickness_mm: float
    markers: tuple[MarkerSpec, ...]
    description: str

    def marker_by_local(self, local_index: int) -> MarkerSpec | None:
        for marker in self.markers:
            if marker.local_index == local_index:
                return marker
        return None


@dataclass
class SheetInstance:
    """One printed page: a layout plus a sheet number, which sets marker IDs."""

    layout: SheetLayout
    sheet_index: int  # 0-based
    markers_by_id: dict[int, MarkerSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        base = LAYOUT_ID_BASE[self.layout.key]
        stride = LAYOUT_STRIDE[self.layout.key]
        self.markers_by_id = {
            base + self.sheet_index * stride + marker.local_index: marker
            for marker in self.layout.markers
        }

    @property
    def page_width_mm(self) -> float:
        return self.layout.page_width_mm

    @property
    def page_height_mm(self) -> float:
        return self.layout.page_height_mm


def _centers(length: float, marker: float, pitch: float, margin_start: float, margin_end: float) -> list[float]:
    usable = length - margin_start - margin_end
    if usable < marker:
        raise ValueError("page is too small for one marker")
    count = int((usable - marker) / pitch) + 1
    count = max(2, count)
    span = marker + (count - 1) * pitch
    leftover = usable - span
    first = margin_start + leftover / 2.0 + marker / 2.0
    return [first + i * pitch for i in range(count)]


def _frame_markers(
    page_w: float,
    page_h: float,
    marker: float,
    pitch: float,
    margin_x: float,
    margin_top: float,
    footer: float,
    interior_every: int = 0,
) -> tuple[MarkerSpec, ...]:
    xs = _centers(page_w, marker, pitch, margin_x, margin_x)
    ys = _centers(page_h, marker, pitch, footer, margin_top)
    specs: list[MarkerSpec] = []
    local = 0
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            border = iy == 0 or iy == len(ys) - 1 or ix == 0 or ix == len(xs) - 1
            interior = False
            if not border and interior_every > 0:
                interior = (ix % interior_every == 0) and (iy % interior_every == 0)
            if not border and not interior:
                continue
            specs.append(MarkerSpec(local, float(x), float(y), float(marker)))
            local += 1
    return tuple(specs)


def _layout(
    key: str,
    label: str,
    page_w: float,
    page_h: float,
    marker: float,
    pitch: float,
    margin_x: float,
    margin_top: float,
    footer: float,
    scale_bar: float,
    description: str,
    interior_every: int = 0,
) -> SheetLayout:
    markers = _frame_markers(
        page_w, page_h, marker, pitch, margin_x, margin_top, footer, interior_every
    )
    stride = LAYOUT_STRIDE[key]
    if len(markers) > stride:
        raise RuntimeError(f"{key} uses {len(markers)} markers but stride is {stride}")
    return SheetLayout(
        key=key,
        label=label,
        page_width_mm=page_w,
        page_height_mm=page_h,
        marker_size_mm=marker,
        footer_height_mm=footer,
        scale_bar_mm=scale_bar,
        scale_bar_x_mm=margin_x + 2.0,
        scale_bar_y_mm=footer * 0.42,
        scale_bar_thickness_mm=max(2.0, marker * 0.12),
        markers=markers,
        description=description,
    )


_INCH = 25.4

LAYOUTS: dict[str, SheetLayout] = {
    "letter": _layout(
        "letter",
        "Letter (8.5 × 11 in)",
        8.5 * _INCH,
        11 * _INCH,
        marker=18.0,
        pitch=32.0,
        margin_x=8.0,
        margin_top=8.0,
        footer=34.0,
        scale_bar=100.0,
        description="Everyday sheet for plates, brackets, gaskets, and templates that fit on a letter page. Print a numbered set and lay the sheets around a larger flat part.",
    ),
    "tabloid": _layout(
        "tabloid",
        "Tabloid (11 × 17 in)",
        11 * _INCH,
        17 * _INCH,
        marker=22.0,
        pitch=48.0,
        margin_x=10.0,
        margin_top=10.0,
        footer=36.0,
        scale_bar=200.0,
        description="Larger shop printer sheet. More open area than letter, still a frame of markers so a part can sit in the middle.",
    ),
    "large": _layout(
        "large",
        "Print-shop board (24 × 36 in)",
        24 * _INCH,
        36 * _INCH,
        marker=36.0,
        pitch=88.0,
        margin_x=16.0,
        margin_top=14.0,
        footer=42.0,
        scale_bar=400.0,
        description="Large-format board for a print shop. Markers frame the sheet and a sparse interior grid remains usable when a part covers the middle. Ask the shop to print at 100% with no scaling.",
        interior_every=2,
    ),
}


def get_layout(key: str) -> SheetLayout:
    if key not in LAYOUTS:
        known = ", ".join(LAYOUTS)
        raise KeyError(f"Unknown sheet layout '{key}'. Choose one of: {known}.")
    return LAYOUTS[key]


def max_sheets(layout_key: str) -> int:
    return LAYOUT_MAX_SHEETS[layout_key]


def make_instance(layout_key: str, sheet_index: int) -> SheetInstance:
    if sheet_index < 0 or sheet_index >= max_sheets(layout_key):
        raise ValueError(
            f"Sheet number {sheet_index + 1} is outside the range for {layout_key} "
            f"(1–{max_sheets(layout_key)})."
        )
    return SheetInstance(get_layout(layout_key), sheet_index)


def lookup_marker_id(marker_id: int) -> tuple[str, int, MarkerSpec] | None:
    """Map a detected ArUco id back to its layout, sheet index, and placement."""
    for key, base in LAYOUT_ID_BASE.items():
        stride = LAYOUT_STRIDE[key]
        count = LAYOUT_MAX_SHEETS[key]
        end = base + count * stride
        if base <= marker_id < end:
            offset = marker_id - base
            sheet_index = offset // stride
            local = offset % stride
            layout = LAYOUTS[key]
            marker = layout.marker_by_local(local)
            if marker is None:
                return None
            return key, sheet_index, marker
    return None


def get_dictionary() -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)


def layout_catalog() -> list[dict]:
    items = []
    for key, layout in LAYOUTS.items():
        items.append(
            {
                "key": key,
                "label": layout.label,
                "page_width_mm": round(layout.page_width_mm, 2),
                "page_height_mm": round(layout.page_height_mm, 2),
                "marker_size_mm": layout.marker_size_mm,
                "marker_count": len(layout.markers),
                "scale_bar_mm": layout.scale_bar_mm,
                "max_sheets": max_sheets(key),
                "description": layout.description,
            }
        )
    return items
