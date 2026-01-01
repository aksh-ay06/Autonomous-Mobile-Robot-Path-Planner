"""
Path Planner wrapper for autonomous mobile robot path planning.
Unified interface for different algorithms + optional smoothing.

Wired extensions:
- blocked_fn: avoid dynamic obstacles, other robots, reservations, etc.
- EnhancedGridMap: uses neighbors_with_cost automatically through search_algorithms.py
- Keeps advanced_algorithms (rrt, rrt_star, prm) if available
- Keeps path_smoothing if available
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, TypeAlias, Union

from .grid_map import GridMap
from .search_algorithms import (
    astar,
    dijkstra,
    manhattan_distance,
    octile_distance,
    euclidean_distance,
)

logger = logging.getLogger(__name__)

Point: TypeAlias = tuple[int, int]
BlockedFn: TypeAlias = Callable[[int, int], bool]
Heuristic: TypeAlias = Callable[[Point, Point], float]


# Optional modules
try:
    from .advanced_algorithms import rrt, rrt_star, prm  # type: ignore
    ADVANCED_ALGORITHMS_AVAILABLE = True
except Exception:
    rrt = rrt_star = prm = None
    ADVANCED_ALGORITHMS_AVAILABLE = False

try:
    from .path_smoothing import smooth_path  # type: ignore
    PATH_SMOOTHING_AVAILABLE = True
except Exception:
    smooth_path = None
    PATH_SMOOTHING_AVAILABLE = False

try:
    from .enhanced_grid import EnhancedGridMap  # type: ignore
except Exception:
    EnhancedGridMap = None  # type: ignore


Algorithm = str  # "dijkstra" | "astar" | "rrt" | "rrt_star" | "prm"


class _GridOverlay:
    """
    Lightweight view over a grid that adds blocked_fn filtering without mutating the grid.
    Keeps neighbors/costs consistent by filtering after neighbor generation.
    """

    def __init__(self, base: GridMap, blocked_fn: Optional[BlockedFn]):
        self._base = base
        self.width = base.width
        self.height = base.height
        self._blocked_fn = blocked_fn

    def is_free(self, x: int, y: int) -> bool:
        if self._blocked_fn and self._blocked_fn(x, y):
            return False
        return self._base.is_free(x, y)

    # 4-neighbors must exist
    def neighbors4(self, x: int, y: int) -> list[Point]:
        if hasattr(self._base, "neighbors4"):
            nbs = self._base.neighbors4(x, y)  # type: ignore[attr-defined]
        else:
            # fallback should not happen with your GridMap
            nbs = []
        return [p for p in nbs if self.is_free(p[0], p[1])]

    # If enhanced grid exists, preserve richer methods
    def neighbors(self, x: int, y: int) -> list[Point]:
        if hasattr(self._base, "neighbors"):
            nbs = self._base.neighbors(x, y)  # type: ignore[attr-defined]
            return [p for p in nbs if self.is_free(p[0], p[1])]
        return self.neighbors4(x, y)

    def neighbors_with_cost(self, x: int, y: int) -> list[tuple[Point, float]]:
        if hasattr(self._base, "neighbors_with_cost"):
            pairs = self._base.neighbors_with_cost(x, y)  # type: ignore[attr-defined]
            return [(p, c) for (p, c) in pairs if self.is_free(p[0], p[1])]
        # fallback to unit costs
        return [(p, 1.0) for p in self.neighbors(x, y)]


@dataclass
class PathPlanner:
    algorithm: Algorithm = "astar"
    heuristic: Heuristic = manhattan_distance
    grid: Optional[Union[GridMap, "EnhancedGridMap"]] = None

    enable_smoothing: bool = False
    smoothing_method: str = "shortcut"
    # Algorithm-specific parameters (RRT/RRT*/PRM/etc.)
    planner_params: dict = field(default_factory=dict)

    # Smoothing-only parameters
    smoothing_params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate algorithm at construction time to mirror change_algorithm behavior
        self.change_algorithm(self.algorithm)

    def set_grid(self, grid: Union[GridMap, "EnhancedGridMap"]) -> None:
        self.grid = grid

    def set_smoothing_params(self, **params) -> None:
        self.smoothing_params.update(params)

    def set_planner_params(self, **params) -> None:
        """Set algorithm-specific parameters (e.g., RRT max_iterations)."""
        self.planner_params.update(params)

    def change_algorithm(self, algorithm: Algorithm) -> None:
        valid = {"dijkstra", "astar"}
        if ADVANCED_ALGORITHMS_AVAILABLE:
            valid |= {"rrt", "rrt_star", "prm"}
        if algorithm not in valid:
            raise ValueError(f"Algorithm must be one of: {sorted(valid)}")
        self.algorithm = algorithm

    def change_heuristic(self, heuristic: Heuristic) -> None:
        self.heuristic = heuristic

    def compute_path(
        self,
        start: Point,
        goal: Point,
        *,
        blocked_fn: Optional[BlockedFn] = None,
    ) -> list[Point]:
        """
        Compute path from start to goal.

        blocked_fn(x,y) lets you avoid:
        - dynamic obstacles
        - other robots
        - reserved cells / constraints
        """
        if self.grid is None:
            raise ValueError("Grid must be set before computing path.")

        overlay = _GridOverlay(self.grid, blocked_fn)

        # grid-based planners
        if self.algorithm == "dijkstra":
            path = dijkstra(start, goal, overlay)

        elif self.algorithm == "astar":
            path = astar(start, goal, overlay, heuristic=self.heuristic)

        # sampling-based planners (best-effort wiring using overlay.is_free)
        elif self.algorithm == "rrt":
            if not ADVANCED_ALGORITHMS_AVAILABLE or rrt is None:
                raise ValueError("RRT not available.")
            # Backward compatible: read from planner_params first, then smoothing_params.
            max_iterations = int(self.planner_params.get("max_iterations", self.smoothing_params.get("max_iterations", 2000)))
            step_size = float(self.planner_params.get("step_size", self.smoothing_params.get("step_size", 1.0)))
            goal_bias = float(self.planner_params.get("goal_bias", self.smoothing_params.get("goal_bias", 0.1)))
            path = rrt(start, goal, overlay, max_iterations, step_size, goal_bias)  # type: ignore[misc]

        elif self.algorithm == "rrt_star":
            if not ADVANCED_ALGORITHMS_AVAILABLE or rrt_star is None:
                raise ValueError("RRT* not available.")
            max_iterations = int(self.planner_params.get("max_iterations", self.smoothing_params.get("max_iterations", 2000)))
            step_size = float(self.planner_params.get("step_size", self.smoothing_params.get("step_size", 1.0)))
            goal_bias = float(self.planner_params.get("goal_bias", self.smoothing_params.get("goal_bias", 0.1)))
            search_radius = float(self.planner_params.get("search_radius", self.smoothing_params.get("search_radius", 2.0)))
            path = rrt_star(start, goal, overlay, max_iterations, step_size, goal_bias, search_radius)  # type: ignore[misc]

        elif self.algorithm == "prm":
            if not ADVANCED_ALGORITHMS_AVAILABLE or prm is None:
                raise ValueError("PRM not available.")
            num_samples = int(self.planner_params.get("num_samples", self.smoothing_params.get("num_samples", 500)))
            connection_radius = float(self.planner_params.get("connection_radius", self.smoothing_params.get("connection_radius", 2.0)))
            path = prm(start, goal, overlay, num_samples, connection_radius)  # type: ignore[misc]

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Normalize: ensure path begins with start if non-empty
        if path and path[0] != start:
            path = [start] + path

        # Apply smoothing only if installed & enabled
        if path and self.enable_smoothing and PATH_SMOOTHING_AVAILABLE and smooth_path is not None:
            try:
                path = smooth_path(path, overlay, self.smoothing_method, **self.smoothing_params)  # type: ignore[misc]
            except Exception as e:
                logger.warning("Path smoothing failed (%s). Returning raw path.", e)

        return path
