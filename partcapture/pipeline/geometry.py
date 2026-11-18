"""Fit lines, arcs, and circles to a measured outline.

Contours from a photo are jagged at the pixel. This module replaces long runs
with the primitive that actually fits, and uses least-squares line intersections
for corners so a plate's width is not sitting on a single noisy pixel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinePrim:
    start: np.ndarray
    end: np.ndarray
    origin: np.ndarray
    direction: np.ndarray  # unit vector start -> end

    def to_json(self) -> dict:
        return {
            "type": "line",
            "x1": float(self.start[0]),
            "y1": float(self.start[1]),
            "x2": float(self.end[0]),
            "y2": float(self.end[1]),
        }


@dataclass
class ArcPrim:
    center: np.ndarray
    radius: float
    start: np.ndarray
    end: np.ndarray
    mid: np.ndarray
    start_angle_deg: float
    end_angle_deg: float  # DXF CCW sweep from start to end

    def to_json(self) -> dict:
        return {
            "type": "arc",
            "cx": float(self.center[0]),
            "cy": float(self.center[1]),
            "r": float(self.radius),
            "a0": float(self.start_angle_deg),
            "a1": float(self.end_angle_deg),
            "x1": float(self.start[0]),
            "y1": float(self.start[1]),
            "x2": float(self.end[0]),
            "y2": float(self.end[1]),
        }


@dataclass
class CirclePrim:
    center: np.ndarray
    radius: float

    def to_json(self) -> dict:
        return {
            "type": "circle",
            "cx": float(self.center[0]),
            "cy": float(self.center[1]),
            "r": float(self.radius),
        }


@dataclass
class FittedOutline:
    outline: list[LinePrim | ArcPrim] = field(default_factory=list)
    holes: list[CirclePrim | list] = field(default_factory=list)
    outer_points_mm: np.ndarray | None = None
    hole_points_mm: list[np.ndarray] = field(default_factory=list)

    def to_json(self) -> dict:
        holes_json = []
        for hole in self.holes:
            if isinstance(hole, CirclePrim):
                holes_json.append(hole.to_json())
            else:
                holes_json.append([p.to_json() for p in hole])
        return {
            "outline": [p.to_json() for p in self.outline],
            "holes": holes_json,
        }


def _as_closed(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) >= 2 and np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
        pts = pts[:-1]
    return pts


def resample_closed(points: np.ndarray, step_mm: float = 0.35) -> np.ndarray:
    pts = _as_closed(points)
    if len(pts) < 3:
        return pts
    loop = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    total = float(seg.sum())
    if total < step_mm * 4:
        return pts
    n = max(12, int(round(total / step_mm)))
    targets = np.linspace(0.0, total, n, endpoint=False)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = np.empty((n, 2), dtype=np.float64)
    for i, target in enumerate(targets):
        idx = int(np.searchsorted(cum, target, side="right") - 1)
        idx = min(max(idx, 0), len(seg) - 1)
        span = seg[idx]
        t = 0.0 if span <= 1e-9 else (target - cum[idx]) / span
        out[i] = loop[idx] * (1.0 - t) + loop[idx + 1] * t
    return out


def fit_circle(points: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic least-squares circle. Returns cx, cy, r, mean absolute radial error."""
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    design = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(pts))])
    target = x * x + y * y
    sol, *_ = np.linalg.lstsq(design, target, rcond=None)
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 1e-8:
        return float(cx), float(cy), 0.0, 1e9
    r = math.sqrt(float(r2))
    resid = np.abs(np.hypot(x - cx, y - cy) - r)
    return float(cx), float(cy), float(r), float(resid.mean())


def fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Total least squares line. Returns origin, unit direction, per-point distances, mean abs."""
    pts = np.asarray(points, dtype=np.float64)
    origin = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - origin, full_matrices=False)
    direction = vt[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / norm
    normal = np.array([-direction[1], direction[0]])
    dist = np.abs((pts - origin) @ normal)
    return origin, direction, dist, float(dist.mean())


def _turn_deg(points: np.ndarray, index: int, k: int) -> float:
    n = len(points)
    a = points[(index - k) % n]
    b = points[index]
    c = points[(index + k) % n]
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _corner_indices(points: np.ndarray, turn_below_deg: float = 150.0) -> list[int]:
    n = len(points)
    if n < 12:
        return []
    k = max(2, n // 50)
    angles = np.array([_turn_deg(points, i, k) for i in range(n)])
    min_sep = max(3, n // 30)
    candidates = np.where(angles < turn_below_deg)[0]
    if len(candidates) == 0:
        return []
    order = candidates[np.argsort(angles[candidates])]
    chosen: list[int] = []
    for idx in order:
        if all(_ring_dist(idx, other, n) >= min_sep for other in chosen):
            chosen.append(int(idx))
    chosen.sort()
    return chosen


def _ring_dist(a: int, b: int, n: int) -> int:
    d = abs(a - b)
    return min(d, n - d)


def _arc_span_deg(points: np.ndarray, cx: float, cy: float) -> float:
    ang = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    ang = np.sort((ang + 2 * math.pi) % (2 * math.pi))
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(ang)
    wrap = ang[0] + 2 * math.pi - ang[-1]
    largest_gap = max(float(gaps.max()) if len(gaps) else 0.0, float(wrap))
    return math.degrees(2 * math.pi - largest_gap)


def _try_full_circle(points: np.ndarray, tol_mm: float) -> CirclePrim | None:
    if len(points) < 8:
        return None
    cx, cy, r, resid = fit_circle(points)
    if r < 0.8 or resid > tol_mm:
        return None
    if _arc_span_deg(points, cx, cy) < 320.0:
        return None
    return CirclePrim(center=np.array([cx, cy]), radius=r)


def _line_prim(points: np.ndarray) -> LinePrim:
    origin, direction, _dist, _mean = fit_line(points)
    # Keep the chain's own endpoints so neighboring primitives still meet.
    # Callers only accept a line when every point is already close to it.
    travel = points[-1] - points[0]
    if float(np.dot(direction, travel)) < 0:
        direction = -direction
    return LinePrim(
        start=np.asarray(points[0], dtype=np.float64).copy(),
        end=np.asarray(points[-1], dtype=np.float64).copy(),
        origin=origin,
        direction=direction,
    )


def _arc_from_points(points: np.ndarray) -> ArcPrim | None:
    cx, cy, r, resid = fit_circle(points)
    if r < 0.8 or not math.isfinite(r):
        return None
    start = points[0]
    end = points[-1]
    mid = points[len(points) // 2]
    a0 = math.degrees(math.atan2(start[1] - cy, start[0] - cx))
    a1 = math.degrees(math.atan2(end[1] - cy, end[0] - cx))
    am = math.degrees(math.atan2(mid[1] - cy, mid[0] - cx))
    if _ccw_contains(a0, a1, am):
        start_a, end_a = a0, a1
        s_pt, e_pt = start, end
    else:
        start_a, end_a = a1, a0
        s_pt, e_pt = end, start
    return ArcPrim(
        center=np.array([cx, cy], dtype=np.float64),
        radius=r,
        start=s_pt.copy(),
        end=e_pt.copy(),
        mid=mid.copy(),
        start_angle_deg=_norm_deg(start_a),
        end_angle_deg=_norm_deg(end_a),
    )


def _norm_deg(angle: float) -> float:
    return float(angle % 360.0)


def _ccw_contains(start: float, end: float, mid: float) -> bool:
    s = _norm_deg(start)
    e = _norm_deg(end)
    m = _norm_deg(mid)
    if s <= e:
        return s <= m <= e
    return m >= s or m <= e


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _fit_open(
    points: np.ndarray,
    line_tol: float,
    arc_tol: float,
    depth: int,
) -> list[LinePrim | ArcPrim]:
    if len(points) < 2:
        return []
    # Path length, not the end-to-end chord: a closed outline is opened by
    # repeating its first point, so the chord is zero while the path is the perimeter.
    path = _path_length(points)
    if len(points) < 4 or path < 0.8:
        if path < 0.25:
            return []
        return [_line_prim(points)]

    _origin, _direction, dist, _line_mean = fit_line(points)
    # Mean error hides shoulders that have already walked onto a round.
    # Max error keeps a straight edge from swallowing the start of an arc.
    # A single jagged pixel is cleaned up later by the collinear merge.
    line_res = float(np.max(dist))
    cx, cy, r, arc_res = fit_circle(points)
    span = _arc_span_deg(points, cx, cy) if r > 0 else 0.0
    arc_ok = arc_res <= arc_tol and 0.8 < r < 8000 and 18.0 <= span <= 320.0

    if line_res <= line_tol and (not arc_ok or line_res <= arc_res * 1.25):
        return [_line_prim(points)]
    if arc_ok and arc_res < line_res * 0.7:
        arc = _arc_from_points(points)
        if arc is not None:
            return [arc]
    # Keep splitting while the chain is not a line or an arc. A depth cap that
    # gives up and emits one chord will cut the inside off a round.
    if len(points) <= 8 or depth >= 28:
        if arc_ok:
            arc = _arc_from_points(points)
            if arc is not None:
                return [arc]
        return [_line_prim(points)]

    split = _interior_split_index(dist)
    if split is None:
        split = len(points) // 2
    left = _fit_open(points[: split + 1], line_tol, arc_tol, depth + 1)
    right = _fit_open(points[split:], line_tol, arc_tol, depth + 1)
    return left + right


def _interior_split_index(dist: np.ndarray) -> int | None:
    """Index farthest from the fitted line, ignoring the endpoints."""
    n = len(dist)
    if n < 6:
        return None
    interior = dist.copy()
    margin = max(2, n // 40)
    interior[:margin] = -1.0
    interior[-margin:] = -1.0
    index = int(np.argmax(interior))
    if interior[index] < 0:
        return None
    return index


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _lines_are_one_edge(first: LinePrim, second: LinePrim, tol: float = 0.18) -> bool:
    """Collinear and the shared joint sits on the chord between the outer ends.

    Angle-only merging walks onto a round and pulls the 'straight' edge inside it.
    """
    d1 = first.end - first.start
    d2 = second.end - second.start
    ang = _angle_between(d1, d2)
    if not (ang < 8.0 or abs(ang - 180.0) < 8.0):
        return False
    chord = second.end - first.start
    length = float(np.linalg.norm(chord))
    if length < 1e-6:
        return False
    # Distance from the joint to the chord through the outer endpoints.
    t = float(np.dot(first.end - first.start, chord) / (length * length))
    projection = first.start + t * chord
    if float(np.linalg.norm(first.end - projection)) > tol:
        return False
    gap = float(np.linalg.norm(first.end - second.start))
    return gap < 1.0


def _merge_adjacent(prims: list[LinePrim | ArcPrim]) -> list[LinePrim | ArcPrim]:
    if len(prims) < 2:
        return prims
    changed = True
    items = list(prims)
    while changed and len(items) >= 2:
        changed = False
        merged: list[LinePrim | ArcPrim] = []
        i = 0
        while i < len(items):
            if i < len(items) - 1 and isinstance(items[i], LinePrim) and isinstance(items[i + 1], LinePrim):
                if _lines_are_one_edge(items[i], items[i + 1]):
                    start = items[i].start
                    end = items[i + 1].end
                    combined = _line_prim(np.vstack([start, end]))
                    combined.start = start.copy()
                    combined.end = end.copy()
                    merged.append(combined)
                    i += 2
                    changed = True
                    continue
            merged.append(items[i])
            i += 1
        if (
            len(merged) >= 2
            and isinstance(merged[0], LinePrim)
            and isinstance(merged[-1], LinePrim)
            and _lines_are_one_edge(merged[-1], merged[0])
        ):
            start = merged[-1].start
            end = merged[0].end
            combined = _line_prim(np.vstack([start, end]))
            combined.start = start.copy()
            combined.end = end.copy()
            merged = [combined] + merged[1:-1]
            changed = True
        items = merged
    return items


def _intersect_lines(a: LinePrim, b: LinePrim) -> np.ndarray | None:
    mat = np.column_stack([a.direction, -b.direction])
    det = float(np.linalg.det(mat))
    if abs(det) < 1e-8:
        return None
    try:
        ab = np.linalg.solve(mat, b.origin - a.origin)
    except np.linalg.LinAlgError:
        return None
    return a.origin + ab[0] * a.direction


def _arc_bows_off_chords(points: np.ndarray, radius: float, tol: float) -> bool:
    """True when the circle pulls more than `tol` away from a straight chord.

    Vertices can sit on a circle while the arc between two distant vertices
    bulges far off the polyline. Each input chord is a straight edge we already
    accepted, so the arc has to stay near every one of them.
    """
    if radius <= 0:
        return True
    for start, end in zip(points[:-1], points[1:]):
        half = 0.5 * float(np.linalg.norm(end - start))
        if half >= radius:
            return True
        sagitta = radius - math.sqrt(radius * radius - half * half)
        if sagitta > tol + 0.05:
            return True
    return False


def _absorb_arc_runs(prims: list[LinePrim | ArcPrim], arc_tol: float) -> list[LinePrim | ArcPrim]:
    """Turn a run of short chords into one arc when a circle fits and a line does not.

    Recursive splitting stops as soon as each tiny chord is straight enough, which
    turns a smooth round with no sharp corner into a polyline. This pass puts the
    arc back.
    """
    if len(prims) < 3:
        return prims
    # Only line runs have this problem; leave existing arcs in place by processing
    # the whole loop as junctions when every primitive is still a line.
    if not all(isinstance(p, LinePrim) for p in prims):
        return prims
    junctions = [prims[0].start]
    for prim in prims:
        junctions.append(prim.end)
    # Drop the repeated closing point if the loop is explicitly closed.
    if np.linalg.norm(junctions[0] - junctions[-1]) < 1e-6:
        junctions = junctions[:-1]
    n = len(junctions)
    if n < 4:
        return prims
    # Start at the longest chord so a round end isn't split across the seam.
    seg_len = [float(np.linalg.norm(junctions[(i + 1) % n] - junctions[i])) for i in range(n)]
    start_at = int(np.argmax(seg_len))
    junctions = junctions[start_at:] + junctions[:start_at]
    used_until = 0
    rebuilt: list[LinePrim | ArcPrim] = []
    i = 0
    while i < n:
        best = None
        # Longest arc starting at i. The loop is cyclic.
        for length in range(n - 1, 2, -1):
            idxs = [(i + k) % n for k in range(length + 1)]
            # Don't swallow the whole outline; a full circle is handled earlier.
            if length >= n - 1:
                continue
            pts = np.array([junctions[k] for k in idxs], dtype=np.float64)
            cx, cy, radius, arc_res = fit_circle(pts)
            radial = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - radius)
            max_radial = float(radial.max())
            _o, _d, _dist, line_res = fit_line(pts)
            span = _arc_span_deg(pts, cx, cy)
            sagitta = radius * (1.0 - math.cos(math.radians(span) / 2.0)) if radius < 1e6 else 0.0
            # A true line fits an enormous circle with tiny radial error. Require a
            # visible bow (sagitta and line residual) before replacing chords with an arc.
            curved = (
                max_radial <= arc_tol
                and line_res >= 0.45
                and sagitta >= 0.6
                and arc_res < line_res * 0.5
                and 30.0 <= span <= 220.0
                and 0.8 < radius < 4000
                and not _arc_bows_off_chords(pts, radius, arc_tol)
            )
            if curved:
                best = (length, pts, cx, cy, radius)
                break
        if best is None:
            j = (i + 1) % n
            rebuilt.append(_line_prim(np.vstack([junctions[i], junctions[j]])))
            i += 1
            used_until += 1
            if used_until > n + 2:
                break
            continue
        length, pts, cx, cy, radius = best
        arc = _arc_from_points(pts)
        if arc is None:
            j = (i + 1) % n
            rebuilt.append(_line_prim(np.vstack([junctions[i], junctions[j]])))
            i += 1
        else:
            rebuilt.append(arc)
            i += length
        used_until += 1
        if used_until > n + 2:
            break
    return rebuilt or prims


def _refine_junctions(prims: list[LinePrim | ArcPrim]) -> list[LinePrim | ArcPrim]:
    """Replace line/line joints with the intersection of the fitted lines."""
    if len(prims) < 2:
        return prims
    n = len(prims)
    junctions: list[np.ndarray | None] = [None] * n
    for i in range(n):
        a = prims[i]
        b = prims[(i + 1) % n]
        if isinstance(a, LinePrim) and isinstance(b, LinePrim):
            hit = _intersect_lines(a, b)
            if hit is not None:
                # Reject intersections that fly away from the recorded joint.
                joint = a.end
                if np.linalg.norm(hit - joint) < 8.0:
                    junctions[i] = hit
    refined: list[LinePrim | ArcPrim] = []
    for i, prim in enumerate(prims):
        start = junctions[(i - 1) % n]
        end = junctions[i]
        if isinstance(prim, LinePrim) and start is not None and end is not None:
            direction = end - start
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
                refined.append(
                    LinePrim(
                        start=start.copy(),
                        end=end.copy(),
                        origin=prim.origin.copy(),
                        direction=direction,
                    )
                )
                continue
        refined.append(prim)
    return refined


def fit_closed_outline(
    points: np.ndarray,
    line_tol: float = 0.22,
    arc_tol: float = 0.22,
) -> list[LinePrim | ArcPrim]:
    pts = resample_closed(points, 0.3)
    if len(pts) < 3:
        return []
    circle = _try_full_circle(pts, arc_tol)
    if circle is not None:
        # Represent a fully circular outline as two semicircle arcs so the
        # outline loop stays closed. Holes use CirclePrim separately.
        cx, cy, r = circle.center[0], circle.center[1], circle.radius
        a = np.array([cx + r, cy])
        b = np.array([cx - r, cy])
        top = np.array([cx, cy + r])
        bot = np.array([cx, cy - r])
        return [
            ArcPrim(circle.center.copy(), r, a, b, top, 0.0, 180.0),
            ArcPrim(circle.center.copy(), r, b, a, bot, 180.0, 360.0),
        ]

    corners = _corner_indices(pts)
    if len(corners) >= 2:
        chains = [_chain_between(pts, start_i, corners[(i + 1) % len(corners)]) for i, start_i in enumerate(corners)]
        prims: list[LinePrim | ArcPrim] = []
        for chain in chains:
            prims.extend(_fit_open(chain, line_tol, arc_tol, 0))
    else:
        # Smooth closed shape (obround, oval). Cut at the two ends of a
        # diameter so each open chain has distinct endpoints.
        i0 = int(np.argmax(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
        i1 = int(np.argmax(np.linalg.norm(pts - pts[i0], axis=1)))
        prims = _fit_open(_chain_between(pts, i0, i1), line_tol, arc_tol, 0)
        prims += _fit_open(_chain_between(pts, i1, i0), line_tol, arc_tol, 0)

    prims = [p for p in prims if _prim_length(p) >= 0.3]
    prims = _merge_adjacent(prims)
    prims = _absorb_arc_runs(prims, arc_tol)
    prims = _refine_junctions(prims)
    prims = _merge_adjacent(prims)
    prims = _replace_tiny_arcs(prims)
    prims = _merge_adjacent(prims)
    prims = [p for p in prims if _prim_length(p) >= 0.3]
    return prims


def _replace_tiny_arcs(prims: list[LinePrim | ArcPrim]) -> list[LinePrim | ArcPrim]:
    """Pixel corners sometimes fit a 1 mm arc. Those are not a real fillet."""
    replaced: list[LinePrim | ArcPrim] = []
    for prim in prims:
        if isinstance(prim, ArcPrim) and prim.radius < 2.0:
            replaced.append(
                LinePrim(
                    start=prim.start.copy(),
                    end=prim.end.copy(),
                    origin=(prim.start + prim.end) / 2.0,
                    direction=_unit(prim.end - prim.start),
                )
            )
        else:
            replaced.append(prim)
    return replaced


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.array([1.0, 0.0])
    return vector / norm


def _chain_between(points: np.ndarray, start: int, end: int) -> np.ndarray:
    if end > start:
        return points[start : end + 1]
    return np.vstack([points[start:], points[: end + 1]])


def _prim_length(prim: LinePrim | ArcPrim) -> float:
    if isinstance(prim, LinePrim):
        return float(np.linalg.norm(prim.end - prim.start))
    return max(prim.radius * math.radians(30.0), 1.0)


def fit_hole(points: np.ndarray, circle_tol: float = 0.28) -> CirclePrim | list:
    pts = resample_closed(points, 0.3)
    circle = _try_full_circle(pts, circle_tol)
    if circle is not None:
        return circle
    # A slightly looser circle test: round holes from a mask are a bit noisy.
    if len(pts) >= 8:
        cx, cy, r, resid = fit_circle(pts)
        radial_std = float(np.std(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))
        if r >= 0.8 and resid <= circle_tol and radial_std <= circle_tol and _arc_span_deg(pts, cx, cy) >= 300:
            return CirclePrim(center=np.array([cx, cy]), radius=r)
    return fit_closed_outline(pts)


def outline_from_json(data: dict) -> FittedOutline:
    """Rebuild primitives saved in a job, so a DXF can be re-exported in either unit."""

    def one(item: dict):
        kind = item["type"]
        if kind == "line":
            start = np.array([item["x1"], item["y1"]], dtype=np.float64)
            end = np.array([item["x2"], item["y2"]], dtype=np.float64)
            direction = end - start
            norm = float(np.linalg.norm(direction)) or 1.0
            return LinePrim(start, end, (start + end) / 2.0, direction / norm)
        if kind == "arc":
            return ArcPrim(
                center=np.array([item["cx"], item["cy"]], dtype=np.float64),
                radius=float(item["r"]),
                start=np.array([item.get("x1", item["cx"]), item.get("y1", item["cy"])], dtype=np.float64),
                end=np.array([item.get("x2", item["cx"]), item.get("y2", item["cy"])], dtype=np.float64),
                mid=np.array([item["cx"], item["cy"]], dtype=np.float64),
                start_angle_deg=float(item["a0"]),
                end_angle_deg=float(item["a1"]),
            )
        if kind == "circle":
            return CirclePrim(np.array([item["cx"], item["cy"]], dtype=np.float64), float(item["r"]))
        raise ValueError(f"Unknown geometry type {kind}")

    holes = []
    for hole in data.get("holes", []):
        if isinstance(hole, dict):
            holes.append(one(hole))
        else:
            holes.append([one(part) for part in hole])
    return FittedOutline(outline=[one(part) for part in data.get("outline", [])], holes=holes)


def fit_outline(outer: np.ndarray, holes: list[np.ndarray]) -> FittedOutline:
    outline = fit_closed_outline(outer)
    fitted_holes = [fit_hole(h) for h in holes]
    return FittedOutline(
        outline=outline,
        holes=fitted_holes,
        outer_points_mm=np.asarray(outer, dtype=np.float64),
        hole_points_mm=[np.asarray(h, dtype=np.float64) for h in holes],
    )


def measure_points(points: np.ndarray) -> dict:
    """Overall length and width from the minimum-area rectangle, plus axis-aligned extents."""
    pts = _as_closed(np.asarray(points, dtype=np.float64))
    if len(pts) < 3:
        return {
            "length_mm": 0.0,
            "width_mm": 0.0,
            "angle_deg": 0.0,
            "center_mm": [0.0, 0.0],
            "aabb_width_mm": 0.0,
            "aabb_height_mm": 0.0,
        }
    # PCA-free min-area rectangle via rotating calipers on the convex hull.
    length, width, angle, center = _min_area_rect(pts)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    return {
        "length_mm": float(length),
        "width_mm": float(width),
        "angle_deg": float(angle),
        "center_mm": [float(center[0]), float(center[1])],
        "aabb_width_mm": float(max_xy[0] - min_xy[0]),
        "aabb_height_mm": float(max_xy[1] - min_xy[1]),
        "min_mm": [float(min_xy[0]), float(min_xy[1])],
        "max_mm": [float(max_xy[0]), float(max_xy[1])],
    }


def _min_area_rect(points: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    hull = _convex_hull(points)
    if len(hull) < 3:
        span = points.max(0) - points.min(0)
        return float(max(span)), float(min(span)), 0.0, points.mean(0)
    best_area = 1e18
    best = (0.0, 0.0, 0.0, points.mean(0))
    n = len(hull)
    for i in range(n):
        edge = hull[(i + 1) % n] - hull[i]
        angle = math.atan2(edge[1], edge[0])
        c = math.cos(-angle)
        s = math.sin(-angle)
        rot = np.array([[c, -s], [s, c]])
        turned = (hull - hull[i]) @ rot.T
        min_xy = turned.min(0)
        max_xy = turned.max(0)
        w = float(max_xy[0] - min_xy[0])
        h = float(max_xy[1] - min_xy[1])
        area = w * h
        if area < best_area:
            best_area = area
            center_r = (min_xy + max_xy) / 2.0
            rot_back = np.array([[c, s], [-s, c]])
            center = hull[i] + rot_back @ center_r
            # `angle` is the direction of the edge we rotated onto +X, which is the
            # width of the rotated frame (w). Length must be that axis only when w is longer.
            if w >= h:
                length, width, length_angle = w, h, angle
            else:
                length, width, length_angle = h, w, angle + math.pi / 2.0
            best = (length, width, math.degrees(length_angle), center)
    return best


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(points, 6), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=np.float64)


def primitive_points(outline: list[LinePrim | ArcPrim], step_deg: float = 4.0) -> np.ndarray:
    """Sample the fitted loop, for previews and extent checks."""
    chunks: list[np.ndarray] = []
    for prim in outline:
        if isinstance(prim, LinePrim):
            chunks.append(np.vstack([prim.start, prim.end]))
        else:
            a0 = math.radians(prim.start_angle_deg)
            a1 = math.radians(prim.end_angle_deg)
            sweep = (a1 - a0) % (2 * math.pi)
            if sweep < 1e-6:
                sweep = 2 * math.pi
            n = max(2, int(math.degrees(sweep) / step_deg))
            angles = a0 + np.linspace(0.0, sweep, n)
            xs = prim.center[0] + prim.radius * np.cos(angles)
            ys = prim.center[1] + prim.radius * np.sin(angles)
            chunks.append(np.column_stack([xs, ys]))
    if not chunks:
        return np.zeros((0, 2))
    return np.vstack(chunks)


def hole_records(fitted: FittedOutline) -> list[dict]:
    records = []
    for hole in fitted.holes:
        if isinstance(hole, CirclePrim):
            records.append(
                {
                    "diameter_mm": hole.radius * 2.0,
                    "cx_mm": float(hole.center[0]),
                    "cy_mm": float(hole.center[1]),
                    "shape": "circle",
                }
            )
        else:
            pts = primitive_points(hole)
            meas = measure_points(pts) if len(pts) else None
            records.append(
                {
                    "diameter_mm": None,
                    "shape": "profile",
                    "length_mm": None if meas is None else meas["length_mm"],
                    "width_mm": None if meas is None else meas["width_mm"],
                }
            )
    return records
