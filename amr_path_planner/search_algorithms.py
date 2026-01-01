"""
Search algorithms for autonomous mobile robot path planning.

Includes:
- A* (spatial)
- Dijkstra (via A* with zero heuristic)
- Space-time A* (reservation-table A*) for multi-robot coordination
"""

from __future__ import annotations

import heapq
import math
from typing import Callable, Dict, Optional, TypeAlias

from .grid_map import GridMap

Point: TypeAlias = tuple[int, int]
Heuristic: TypeAlias = Callable[[Point, Point], float]
State: TypeAlias = tuple[int, int, int]  # (x, y, t)


def manhattan_distance(a: Point, b: Point) -> float:
    """Manhattan distance for 4-connected grids with unit step cost."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean_distance(a: Point, b: Point) -> float:
    """Euclidean distance (admissible for 8-connected grids with unit/diagonal costs)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev_distance(a: Point, b: Point) -> float:
    """Chebyshev distance (useful for 8-connected grids with unit cost for all moves)."""
    return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))


def octile_distance(a: Point, b: Point) -> float:
    """Octile distance for 8-connected grids (cost 1 for cardinal, sqrt(2) for diagonal)."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dmin = min(dx, dy)
    dmax = max(dx, dy)
    return (dmax - dmin) + math.sqrt(2.0) * dmin


def _reconstruct_path(came_from: Dict[Point, Optional[Point]], goal: Point) -> list[Point]:
    path: list[Point] = []
    cur: Optional[Point] = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


def dijkstra(start: Point, goal: Point, grid: GridMap) -> list[Point]:
    """Dijkstra on a grid with unit edge costs."""
    return astar(start, goal, grid, heuristic=lambda _a, _b: 0.0)


def astar(start: Point, goal: Point, grid: GridMap, heuristic: Heuristic = manhattan_distance) -> list[Point]:
    """A* on a grid.

    Behavior:
    - If the grid exposes neighbors_with_cost(x,y) -> List[(neighbor, cost)], that API is used.
      This enables 8-connected / custom movement and terrain costs (e.g., EnhancedGridMap).
    - Otherwise, falls back to neighbors4(x,y) with unit step cost.

    Returns an optimal path if reachable (given an admissible heuristic), else [].
    """
    if not grid.is_free(*start) or not grid.is_free(*goal):
        return []

    # heap entries: (f_score, tie_breaker, node)
    frontier: list[tuple[float, int, Point]] = []
    push_id = 0

    came_from: Dict[Point, Optional[Point]] = {start: None}
    g_score: Dict[Point, float] = {start: 0.0}
    f_start = heuristic(start, goal)
    f_score: Dict[Point, float] = {start: f_start}

    heapq.heappush(frontier, (f_start, push_id, start))

    closed: set[Point] = set()

    while frontier:
        current_f, _, current = heapq.heappop(frontier)

        # Skip stale heap entries
        if current_f != f_score.get(current, float("inf")):
            continue

        if current == goal:
            return _reconstruct_path(came_from, goal)

        if current in closed:
            continue
        closed.add(current)

        # Prefer cost-aware neighbor expansion when available.
        if hasattr(grid, "neighbors_with_cost"):
            # type: ignore[attr-defined]
            nbrs = grid.neighbors_with_cost(current[0], current[1])  # (Point, cost)
            neighbor_cost_pairs = [(p, float(c)) for (p, c) in nbrs]
        else:
            neighbor_cost_pairs = [(p, 1.0) for p in grid.neighbors4(*current)]

        for neighbor, step_cost in neighbor_cost_pairs:
            tentative_g = g_score[current] + step_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                f_score[neighbor] = f
                push_id += 1
                heapq.heappush(frontier, (f, push_id, neighbor))

    return []


# ----------------------------
# Space-time A* (Reservation-table A*)
# ----------------------------

def astar_space_time(
    start: Point,
    goal: Point,
    grid: GridMap,
    reserved_v: Dict[tuple[int, Point], int],
    reserved_e: Dict[tuple[int, Point, Point], int],
    robot_id: int,
    horizon: int = 15,
    heuristic: Heuristic = manhattan_distance,
    allow_wait: bool = True,
) -> list[Point]:
    """
    Space-time A* over (cell, time) with reservation tables.

    Reservations:
    - Vertex: reserved_v[(t, cell)] blocks occupying cell at time t
    - Edge: reserved_e[(t, u, v)] blocks traversing u->v between t and t+1

    Returns:
    - Spatial path [p0, p1, ...] aligned to time steps, with length <= horizon+1
    - [] if no path found within horizon

    Notes:
    - This is the correct wiring for multi-robot coordination with time-indexed reservations.
    """
    if not grid.is_free(*start) or not grid.is_free(*goal):
        return []

    def vertex_blocked(t: int, cell: Point) -> bool:
        owner = reserved_v.get((t, cell))
        return owner is not None and owner != robot_id

    def edge_blocked(t: int, u: Point, v: Point) -> bool:
        owner = reserved_e.get((t, u, v))
        return owner is not None and owner != robot_id

    # If our start cell is reserved by someone else at t=0, planning fails.
    if vertex_blocked(0, start):
        return []

    start_state: State = (start[0], start[1], 0)

    frontier: list[tuple[float, int, State]] = []
    push_id = 0

    came_from: Dict[State, Optional[State]] = {start_state: None}
    g_score: Dict[State, float] = {start_state: 0.0}

    f0 = heuristic(start, goal)
    f_score: Dict[State, float] = {start_state: f0}

    heapq.heappush(frontier, (f0, push_id, start_state))

    while frontier:
        cur_f, _, cur = heapq.heappop(frontier)

        # stale
        if cur_f != f_score.get(cur, float("inf")):
            continue

        x, y, t = cur
        cur_cell: Point = (x, y)

        # goal reached
        if cur_cell == goal:
            # reconstruct state path -> spatial path
            out: list[Point] = []
            s: Optional[State] = cur
            while s is not None:
                out.append((s[0], s[1]))
                s = came_from[s]
            out.reverse()
            return out

        if t >= horizon:
            continue  # don't expand beyond horizon

        # candidate moves
        neighbors = grid.neighbors4(x, y)
        if allow_wait:
            neighbors.append(cur_cell)  # wait action

        nt = t + 1

        for nxt_cell in neighbors:
            nx, ny = nxt_cell
            nxt_state: State = (nx, ny, nt)

            # vertex reservation at next time
            if vertex_blocked(nt, nxt_cell):
                continue

            # edge reservation for move
            if nxt_cell != cur_cell:
                if edge_blocked(t, cur_cell, nxt_cell):
                    continue
                # also prevent swaps if opposite edge is reserved
                if edge_blocked(t, nxt_cell, cur_cell):
                    continue

            tentative_g = g_score[cur] + 1.0

            if tentative_g < g_score.get(nxt_state, float("inf")):
                came_from[nxt_state] = cur
                g_score[nxt_state] = tentative_g
                fn = tentative_g + heuristic(nxt_cell, goal)
                f_score[nxt_state] = fn
                push_id += 1
                heapq.heappush(frontier, (fn, push_id, nxt_state))

    return []
