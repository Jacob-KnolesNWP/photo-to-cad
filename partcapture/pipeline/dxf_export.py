"""DXF export in real millimetres or inches, 1:1 for SOLIDWORKS."""

from __future__ import annotations

from pathlib import Path

import ezdxf
from ezdxf import units

from partcapture.pipeline.geometry import ArcPrim, CirclePrim, FittedOutline, LinePrim


def _scale_for(unit: str) -> float:
    if unit == "mm":
        return 1.0
    if unit in {"in", "inch", "inches"}:
        return 1.0 / 25.4
    raise ValueError(f"Unsupported unit '{unit}'. Use mm or in.")


def _ezdxf_units(unit: str):
    if unit == "mm":
        return units.MM
    return units.IN


def write_dxf(
    fitted: FittedOutline,
    path: str | Path,
    unit: str = "mm",
    note: str | None = None,
) -> None:
    unit = "in" if unit in {"in", "inch", "inches"} else "mm"
    scale = _scale_for(unit)
    doc = ezdxf.new("R2010")
    doc.units = _ezdxf_units(unit)
    doc.layers.add("OUTLINE", color=7)
    doc.layers.add("HOLES", color=4)
    doc.layers.add("NOTES", color=8)
    msp = doc.modelspace()
    for prim in fitted.outline:
        _add_prim(msp, prim, scale, "OUTLINE")
    for hole in fitted.holes:
        if isinstance(hole, CirclePrim):
            msp.add_circle(
                (hole.center[0] * scale, hole.center[1] * scale),
                hole.radius * scale,
                dxfattribs={"layer": "HOLES"},
            )
        elif isinstance(hole, list):
            for prim in hole:
                _add_prim(msp, prim, scale, "HOLES")
    if note:
        msp.add_text(
            note,
            height=2.5 * scale,
            dxfattribs={"layer": "NOTES"},
        ).set_placement((0, -8 * scale))
    doc.saveas(str(path))


def _add_prim(msp, prim: LinePrim | ArcPrim, scale: float, layer: str) -> None:
    if isinstance(prim, LinePrim):
        msp.add_line(
            (prim.start[0] * scale, prim.start[1] * scale),
            (prim.end[0] * scale, prim.end[1] * scale),
            dxfattribs={"layer": layer},
        )
    elif isinstance(prim, ArcPrim):
        msp.add_arc(
            (prim.center[0] * scale, prim.center[1] * scale),
            prim.radius * scale,
            prim.start_angle_deg,
            prim.end_angle_deg,
            dxfattribs={"layer": layer},
        )


def dxf_extents_mm(path: str | Path, unit: str = "mm") -> tuple[float, float]:
    """Axis-aligned width and height of model geometry, converted back to mm."""
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    xs: list[float] = []
    ys: list[float] = []
    for entity in msp:
        if entity.dxftype() == "LINE":
            xs.extend([entity.dxf.start.x, entity.dxf.end.x])
            ys.extend([entity.dxf.start.y, entity.dxf.end.y])
        elif entity.dxftype() == "CIRCLE":
            c = entity.dxf.center
            r = entity.dxf.radius
            xs.extend([c.x - r, c.x + r])
            ys.extend([c.y - r, c.y + r])
        elif entity.dxftype() == "ARC":
            # Sample the arc so the extent includes the bulge, not just the ends.
            import math

            c = entity.dxf.center
            r = entity.dxf.radius
            a0 = math.radians(entity.dxf.start_angle)
            a1 = math.radians(entity.dxf.end_angle)
            sweep = (a1 - a0) % (2 * math.pi)
            for i in range(25):
                a = a0 + sweep * i / 24
                xs.append(c.x + r * math.cos(a))
                ys.append(c.y + r * math.sin(a))
    if not xs:
        return 0.0, 0.0
    to_mm = 25.4 if unit in {"in", "inch", "inches"} else 1.0
    return (max(xs) - min(xs)) * to_mm, (max(ys) - min(ys)) * to_mm
