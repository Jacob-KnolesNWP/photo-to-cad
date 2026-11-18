"""Primitive fitting on analytic outlines, with no camera in the loop."""

import math

import numpy as np

from partcapture.pipeline.geometry import (
    ArcPrim,
    CirclePrim,
    LinePrim,
    fit_hole,
    fit_outline,
    measure_points,
    primitive_points,
)
from partcapture.synthetic import circle, obround, rectangle


def _sample_closed(points: np.ndarray, step: float = 0.4) -> np.ndarray:
    loop = np.vstack([points, points[0]])
    out = [loop[0]]
    for a, b in zip(loop[:-1], loop[1:]):
        dist = np.linalg.norm(b - a)
        n = max(1, int(dist / step))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return np.asarray(out[:-1], dtype=np.float64)


def test_rectangle_fits_four_lines():
    outer = _sample_closed(rectangle(0, 0, 120, 80))
    fitted = fit_outline(outer, [])
    lines = [p for p in fitted.outline if isinstance(p, LinePrim)]
    arcs = [p for p in fitted.outline if isinstance(p, ArcPrim)]
    assert len(arcs) == 0
    assert len(lines) == 4
    meas = measure_points(primitive_points(fitted.outline))
    assert abs(meas["length_mm"] - 120) < 0.15
    assert abs(meas["width_mm"] - 80) < 0.15


def test_circle_hole_becomes_a_circle():
    outer = _sample_closed(rectangle(0, 0, 100, 60))
    hole = circle(40, 30, 10, n=120)
    fitted = fit_outline(outer, [hole])
    assert len(fitted.holes) == 1
    assert isinstance(fitted.holes[0], CirclePrim)
    assert abs(fitted.holes[0].radius * 2 - 20) < 0.15


def test_obround_has_lines_and_arcs():
    shape = obround(0, 0, 120, 40, n_cap=64)
    fitted = fit_outline(shape, [])
    lines = [p for p in fitted.outline if isinstance(p, LinePrim)]
    arcs = [p for p in fitted.outline if isinstance(p, ArcPrim)]
    assert lines, "expected straight sides"
    assert arcs, "expected round ends"
    meas = measure_points(primitive_points(fitted.outline))
    assert abs(meas["length_mm"] - 120) < 0.4
    assert abs(meas["width_mm"] - 40) < 0.4
    radii = [a.radius for a in arcs]
    assert abs(float(np.median(radii)) - 20) < 0.5


def test_fit_hole_rejects_a_slot():
    # A long slot should not collapse to one circle.
    slot = obround(0, 0, 40, 10, n_cap=24)
    fitted = fit_hole(slot)
    assert not isinstance(fitted, CirclePrim)


def test_length_axis_is_the_long_side():
    meas = measure_points(rectangle(0, 0, 50, 20))
    # Long side is along +X, so the length angle is ~0 or ~180.
    angle = meas["angle_deg"] % 180
    assert angle < 5 or angle > 175
    assert abs(meas["length_mm"] - 50) < 1e-6
    assert abs(meas["width_mm"] - 20) < 1e-6
