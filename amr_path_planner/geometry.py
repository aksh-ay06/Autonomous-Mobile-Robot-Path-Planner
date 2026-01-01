"""Geometry and collision helpers.

These utilities are shared by:
- sampling-based planners (RRT/RRT*/PRM)
- path smoothing

Keeping them in one place avoids duplicated Bresenham / collision logic.
"""

from __future__ import annotations

import math
from typing import TypeAlias

from .grid_map import GridMap

Point: TypeAlias = tuple[int, int]
FPoint: TypeAlias = tuple[float, float]


def euclidean_distance(a: FPoint, b: FPoint) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bresenham_cells(a: Point, b: Point) -> list[Point]:
    """Return grid cells intersected by the line from a to b (Bresenham)."""
    x0, y0 = a
    x1, y1 = b
    cells: list[Point] = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    x, y = x0, y0
    if dx >= dy:
        err = dx // 2
        while x != x1:
            cells.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
        cells.append((x1, y1))
    else:
        err = dy // 2
        while y != y1:
            cells.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
        cells.append((x1, y1))

    return cells


def line_collision_check(start: FPoint, end: FPoint, grid: GridMap) -> bool:
    """Collision check for a line segment in a grid.

    Continuous coordinates are snapped to nearest integer cells via round() and
    the segment is checked by traversing intersected cells using Bresenham.

    Returns True if the segment intersects an occupied/out-of-bounds cell.
    """
    a = (int(round(start[0])), int(round(start[1])))
    b = (int(round(end[0])), int(round(end[1])))
    for cx, cy in bresenham_cells(a, b):
        if not grid.is_free(cx, cy):
            return True
    return False
