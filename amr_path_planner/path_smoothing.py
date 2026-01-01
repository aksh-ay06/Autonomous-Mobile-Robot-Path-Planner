"""
Path smoothing and optimization algorithms for autonomous mobile robot navigation.

All methods are wired through smooth_path():
- shortcut
- adaptive
- douglas_peucker
- bezier  (implemented via Chaikin corner-cutting; smooth curve-like)
- spline  (resample + optional gaussian smoothing; SciPy optional)

By default, smooth_path returns List[(int,int)] suitable for GridMap planning.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional, TypeAlias

import numpy as np

from .grid_map import GridMap
from .geometry import euclidean_distance, line_collision_check

Point: TypeAlias = tuple[int, int]
FPoint: TypeAlias = tuple[float, float]


def _path_is_collision_free(path: List[Point], grid: GridMap) -> bool:
    if len(path) <= 1:
        return True
    for i in range(len(path) - 1):
        a = (float(path[i][0]), float(path[i][1]))
        b = (float(path[i + 1][0]), float(path[i + 1][1]))
        if line_collision_check(a, b, grid):
            return False
    return True


def _dedupe_consecutive(path: List[Point]) -> List[Point]:
    if not path:
        return []
    out = [path[0]]
    for p in path[1:]:
        if p != out[-1]:
            out.append(p)
    return out


# ----------------------------
# Smoothing methods
# ----------------------------

def shortcut_smoothing(path: List[Point], grid: GridMap, max_iterations: int = 100, seed: Optional[int] = None) -> List[Point]:
    """Random shortcut smoothing (collision-aware)."""
    if len(path) <= 2:
        return path.copy()

    rng = np.random.default_rng(seed)
    pts: list[FPoint] = [(float(x), float(y)) for x, y in path]

    for _ in range(max_iterations):
        if len(pts) <= 2:
            break
        i = int(rng.integers(0, len(pts) - 2))
        j = int(rng.integers(i + 2, len(pts)))
        if not line_collision_check(pts[i], pts[j], grid):
            pts = pts[: i + 1] + pts[j:]

    out = [(int(round(x)), int(round(y))) for x, y in pts]
    return _dedupe_consecutive(out)


def calculate_curvature(path: List[Point], index: int) -> float:
    """Turning angle magnitude (0 for straight, up to pi for U-turn)."""
    if index <= 0 or index >= len(path) - 1:
        return 0.0

    p1 = np.array(path[index - 1], dtype=float)
    p2 = np.array(path[index], dtype=float)
    p3 = np.array(path[index + 1], dtype=float)

    v1 = p2 - p1
    v2 = p3 - p2

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0

    cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    return angle


def adaptive_smoothing(
    path: List[Point],
    grid: Optional[GridMap] = None,
    curvature_threshold: float = 0.5,
    lookahead: int = 10,
) -> List[Point]:
    """Greedy smoothing while preserving sharp turns; grid optional (assume free space when absent)."""
    if len(path) <= 2:
        return path.copy()

    def segment_blocked(a: Point, b: Point) -> bool:
        if grid is None:
            return False
        return line_collision_check((float(a[0]), float(a[1])), (float(b[0]), float(b[1])), grid)

    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        cur = path[i]
        max_reach = i + 1

        for j in range(i + 2, min(i + lookahead, len(path))):
            if not segment_blocked(cur, path[j]):
                max_reach = j
            else:
                break

        # preserve sharp turns inside the skipped region
        if max_reach > i + 2:
            for k in range(i + 1, max_reach):
                if calculate_curvature(path, k) > curvature_threshold:
                    max_reach = k
                    break

        smoothed.append(path[max_reach])
        i = max_reach

    if smoothed[-1] != path[-1]:
        smoothed.append(path[-1])
    return _dedupe_consecutive(smoothed)


def douglas_peucker_smoothing(path: List[Point], epsilon: float = 1.0) -> List[Point]:
    """Pure geometric simplification (NOT obstacle-aware by itself)."""
    if len(path) <= 2:
        return path.copy()

    def point_line_distance(p: Point, a: Point, b: Point) -> float:
        x0, y0 = p
        x1, y1 = a
        x2, y2 = b
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = math.hypot(y2 - y1, x2 - x1)
        return num / (den + 1e-9)

    def rec(points: List[Point]) -> List[Point]:
        if len(points) <= 2:
            return points
        max_d = 0.0
        idx = 0
        for i in range(1, len(points) - 1):
            d = point_line_distance(points[i], points[0], points[-1])
            if d > max_d:
                max_d = d
                idx = i
        if max_d > epsilon:
            left = rec(points[: idx + 1])
            right = rec(points[idx:])
            return left[:-1] + right
        return [points[0], points[-1]]

    return _dedupe_consecutive(rec(path))


def bezier_smoothing(path: List[Point], iterations: int = 2, num_points: Optional[int] = None) -> List[FPoint]:
    """
    Curve-like smoothing using Chaikin's corner-cutting algorithm.
    (Common practical substitute for Bezier-like smoothing on polylines.)
    Returns float points; optionally resamples to `num_points`.
    """
    if len(path) <= 2:
        return [(float(x), float(y)) for x, y in path]

    pts = [(float(x), float(y)) for x, y in path]
    for _ in range(max(0, iterations)):
        new_pts: list[FPoint] = [pts[0]]
        for i in range(len(pts) - 1):
            p0 = np.array(pts[i], dtype=float)
            p1 = np.array(pts[i + 1], dtype=float)
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.append((float(q[0]), float(q[1])))
            new_pts.append((float(r[0]), float(r[1])))
        new_pts.append(pts[-1])
        pts = new_pts

    if num_points is not None and num_points > 1:
        # Resample uniformly along cumulative arc length to the requested count.
        arr = np.array(pts, dtype=float)
        seg_lengths = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total = cum[-1]
        if total <= 0:
            pts = [pts[0]] * num_points
        else:
            samples = np.linspace(0.0, total, num_points)
            xs = np.interp(samples, cum, arr[:, 0])
            ys = np.interp(samples, cum, arr[:, 1])
            pts = [(float(x), float(y)) for x, y in zip(xs, ys)]

    # Preserve exact endpoints
    pts[0] = (float(path[0][0]), float(path[0][1]))
    pts[-1] = (float(path[-1][0]), float(path[-1][1]))
    return pts


def spline_smoothing(
    path: List[Point],
    smoothing_factor: float = 0.5,
    upsample: int = 5,
    num_points: Optional[int] = None,
) -> List[FPoint]:
    """
    Resample path and apply optional smoothing (SciPy if available, else moving average).
    Returns float points; if num_points is provided, output length matches exactly.
    """
    if len(path) <= 2:
        return [(float(x), float(y)) for x, y in path]

    arr = np.array(path, dtype=float)
    t = np.linspace(0, 1, len(arr))
    target_len = num_points if num_points is not None else len(arr) * max(2, upsample)
    target_len = max(int(target_len), 2)
    t2 = np.linspace(0, 1, target_len)

    xs = np.interp(t2, t, arr[:, 0])
    ys = np.interp(t2, t, arr[:, 1])

    # smooth
    try:
        from scipy import ndimage  # noqa: F401
        sigma = max(0.0, float(smoothing_factor)) * 2.0
        if sigma > 0:
            xs = ndimage.gaussian_filter1d(xs, sigma)
            ys = ndimage.gaussian_filter1d(ys, sigma)
    except ImportError:
        # fallback: moving average
        win = max(1, int(max(0.0, float(smoothing_factor)) * 7))
        if win > 1:
            kernel = np.ones(win, dtype=float) / win
            xs = np.convolve(xs, kernel, mode="same")
            ys = np.convolve(ys, kernel, mode="same")

    out = [(float(x), float(y)) for x, y in zip(xs, ys)]
    # Preserve endpoints exactly to satisfy tests
    out[0] = (float(path[0][0]), float(path[0][1]))
    out[-1] = (float(path[-1][0]), float(path[-1][1]))
    return out


# ----------------------------
# Main wiring + utilities
# ----------------------------

def smooth_path(
    path: List[Point],
    grid: GridMap,
    method: str = "shortcut",
    collision_safe: bool = True,
    **kwargs,
) -> List[Point]:
    """
    Apply path smoothing using specified method.

    Methods:
      - "shortcut"
      - "adaptive"
      - "douglas_peucker"
      - "bezier"
      - "spline"

    By default returns grid integer points. If collision_safe=True, will fall back
    to the original path if the smoothed path introduces obstacle collisions.
    """
    if len(path) <= 2:
        return path.copy()

    method = method.lower().strip()

    if method == "shortcut":
        max_iterations = int(kwargs.get("max_iterations", 100))
        seed = kwargs.get("seed", None)
        out = shortcut_smoothing(path, grid, max_iterations=max_iterations, seed=seed)

    elif method == "adaptive":
        curvature_threshold = float(kwargs.get("curvature_threshold", 0.5))
        lookahead = int(kwargs.get("lookahead", 10))
        out = adaptive_smoothing(path, grid, curvature_threshold=curvature_threshold, lookahead=lookahead)

    elif method == "douglas_peucker":
        epsilon = float(kwargs.get("epsilon", 1.0))
        out = douglas_peucker_smoothing(path, epsilon=epsilon)

    elif method == "bezier":
        iterations = int(kwargs.get("iterations", 2))
        pts = bezier_smoothing(path, iterations=iterations)
        out = _dedupe_consecutive([(int(round(x)), int(round(y))) for x, y in pts])

    elif method == "spline":
        smoothing_factor = float(kwargs.get("smoothing_factor", 0.5))
        upsample = int(kwargs.get("upsample", 5))
        pts = spline_smoothing(path, smoothing_factor=smoothing_factor, upsample=upsample)
        out = _dedupe_consecutive([(int(round(x)), int(round(y))) for x, y in pts])

    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    # Ensure endpoints preserved
    if out and out[0] != path[0]:
        out[0] = path[0]
    if out and out[-1] != path[-1]:
        out[-1] = path[-1]

    # Collision safety: don’t allow smoothing to create collisions
    if collision_safe and not _path_is_collision_free(out, grid):
        return path.copy()

    return out


def path_length(path: List[Point]) -> float:
    if len(path) <= 1:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        total += euclidean_distance((float(path[i][0]), float(path[i][1])),
                                    (float(path[i + 1][0]), float(path[i + 1][1])))
    return float(total)


def path_curvature_analysis(path: List[Point]) -> dict:
    if len(path) <= 2:
        return {
            "mean_curvature": 0.0,
            "max_curvature": 0.0,
            "sharp_turns": 0,
            "total_length": path_length(path),
            "num_waypoints": len(path),
        }

    curvatures = [calculate_curvature(path, i) for i in range(1, len(path) - 1)]
    sharp_turn_threshold = math.pi / 3  # 60 degrees
    sharp_turns = sum(1 for c in curvatures if c > sharp_turn_threshold)

    return {
        "mean_curvature": float(np.mean(curvatures)) if curvatures else 0.0,
        "max_curvature": float(max(curvatures)) if curvatures else 0.0,
        "sharp_turns": int(sharp_turns),
        "total_length": float(path_length(path)),
        "num_waypoints": int(len(path)),
    }


def analyze_path_smoothness(path: List[Point]) -> dict:
    if len(path) < 3:
        return {
            "path_length": 0.0,
            "total_angle_change": 0.0,
            "avg_angle_change": 0.0,
            "max_angle_change": 0.0,
            "num_turns": 0,
            "avg_segment_length": 0.0,
        }

    segment_lengths: list[float] = []
    angles: list[float] = []
    total_len = 0.0

    for i in range(len(path) - 1):
        p1 = np.array(path[i], dtype=float)
        p2 = np.array(path[i + 1], dtype=float)
        seg = float(np.linalg.norm(p2 - p1))
        total_len += seg
        segment_lengths.append(seg)

        if i < len(path) - 2:
            p3 = np.array(path[i + 2], dtype=float)
            v1 = p2 - p1
            v2 = p3 - p2
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-9 and n2 > 1e-9:
                cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                angles.append(float(np.arccos(cos_a)))
            else:
                angles.append(0.0)

    total_angle_change = float(np.sum(angles)) if angles else 0.0
    avg_angle_change = float(np.mean(angles)) if angles else 0.0
    max_angle_change = float(np.max(angles)) if angles else 0.0
    num_turns = int(sum(1 for a in angles if a > 0.017))  # ~1 degree

    return {
        "path_length": float(total_len),
        "total_angle_change": total_angle_change,
        "avg_angle_change": avg_angle_change,
        "max_angle_change": max_angle_change,
        "num_turns": num_turns,
        "avg_segment_length": float(np.mean(segment_lengths)) if segment_lengths else 0.0,
    }
