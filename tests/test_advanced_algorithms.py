"""
Test suite for advanced path planning algorithms (RRT, RRT*, PRM).

UPDATED to match the *actual* implementation in this repo:
- Uses function-based planners: rrt, rrt_star, prm (no RRTPlanner/RRTStarPlanner/PRMPlanner classes).
- Node is optional; tested only if present and if it matches the (position,parent,cost) signature.
- Removes reliance on private methods (_get_random_point, _get_nearest_node, etc.) that don't exist.
- Handles probabilistic planners correctly using retries for "should succeed" cases.
- Adds stronger path validity checks:
  - in-bounds + free cells
  - collision-free edges via conservative Bresenham line-of-sight checks
- Fixes old GridMap API usage: prefers is_free(x,y) over is_obstacle(x,y).
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from typing import Callable, List, Optional, Tuple

# Add the parent directory to the path to import amr_path_planner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_path_planner.grid_map import GridMap

Point = Tuple[int, int]

# Advanced algorithms are optional in some installs/builds
try:
    from amr_path_planner.advanced_algorithms import Node, rrt, rrt_star, prm  # type: ignore

    ADVANCED_AVAILABLE = True
except Exception:
    ADVANCED_AVAILABLE = False
    Node = None  # type: ignore
    rrt = None  # type: ignore
    rrt_star = None  # type: ignore
    prm = None  # type: ignore


def _in_bounds(grid: GridMap, p: Point) -> bool:
    x, y = p
    return 0 <= x < grid.width and 0 <= y < grid.height


def _bresenham_line(a: Point, b: Point) -> List[Point]:
    """Discrete line between a and b (inclusive) for collision checking."""
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    pts: List[Point] = []
    while True:
        pts.append((x0, y0))
        if (x0, y0) == (x1, y1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return pts


def _edge_is_free(grid: GridMap, a: Point, b: Point) -> bool:
    """Conservative LOS check: all cells along the discrete line must be free."""
    for x, y in _bresenham_line(a, b):
        if not grid.is_free(x, y):
            return False
    return True


def _assert_path_valid(testcase: unittest.TestCase, grid: GridMap, start: Point, goal: Point, path: List[Point]) -> None:
    testcase.assertIsInstance(path, list)

    if not path:
        return

    testcase.assertEqual(path[0], start, "Path must start at start")
    testcase.assertEqual(path[-1], goal, "Path must end at goal")

    for p in path:
        testcase.assertTrue(_in_bounds(grid, p), f"Point out of bounds: {p}")
        testcase.assertTrue(grid.is_free(p[0], p[1]), f"Point in obstacle: {p}")

    for i in range(len(path) - 1):
        testcase.assertTrue(_edge_is_free(grid, path[i], path[i + 1]), f"Edge crosses obstacle: {path[i]}->{path[i+1]}")


def _run_with_retries(fn: Callable[[], List[Point]], attempts: int) -> List[Point]:
    """Run a probabilistic planner multiple times; return first non-empty path."""
    for _ in range(attempts):
        p = fn()
        if p:
            return p
    return []


@unittest.skipUnless(ADVANCED_AVAILABLE, "Advanced algorithms not available")
class TestAdvancedAlgorithmsFunctions(unittest.TestCase):
    """Test the function-based advanced algorithms: rrt, rrt_star, prm."""

    def setUp(self) -> None:
        self.grid = GridMap(10, 10)
        # Add a simple obstacle line with a gap
        for i in range(3, 7):
            if i != 5:
                self.grid.add_obstacle(i, 5)

        self.start: Point = (0, 0)
        self.goal: Point = (9, 9)

        self.assertTrue(self.grid.is_free(*self.start))
        self.assertTrue(self.grid.is_free(*self.goal))

    def test_node_optional(self) -> None:
        """Node is optional; validate basic behavior only if it's present and compatible."""
        if Node is None:
            self.skipTest("Node not exposed in this build")

        # Your repo's Node (from updated advanced_algorithms) uses Node(position, parent=None)
        # and has .position, .parent, .cost and maybe .children/add_child.
        try:
            n = Node((5, 10))
        except Exception as e:
            self.skipTest(f"Node signature mismatch in this build: {e}")

        self.assertEqual(getattr(n, "position", None), (5, 10))
        self.assertTrue(hasattr(n, "parent"))
        self.assertTrue(hasattr(n, "cost"))

    def test_rrt_returns_valid_path_if_found(self) -> None:
        self.assertIsNotNone(rrt)
        path = rrt(self.start, self.goal, self.grid, max_iterations=1000, step_size=1.0, goal_bias=0.1)
        _assert_path_valid(self, self.grid, self.start, self.goal, path)

    def test_rrt_star_returns_valid_path_if_found(self) -> None:
        self.assertIsNotNone(rrt_star)
        path = rrt_star(self.start, self.goal, self.grid, max_iterations=1000, step_size=1.0, goal_bias=0.1, search_radius=3.0)
        _assert_path_valid(self, self.grid, self.start, self.goal, path)

    def test_prm_returns_valid_path_if_found(self) -> None:
        self.assertIsNotNone(prm)
        path = prm(self.start, self.goal, self.grid, num_samples=200, connection_radius=3.0)
        _assert_path_valid(self, self.grid, self.start, self.goal, path)

    def test_algorithms_find_path_in_open_grid(self) -> None:
        """In an obstacle-free grid, all algorithms should find a path (with retries)."""
        g = GridMap(10, 10)
        start: Point = (1, 1)
        goal: Point = (8, 8)

        algs: List[Tuple[str, Callable[[], List[Point]]]] = []

        if rrt is not None:
            algs.append(("RRT", lambda: rrt(start, goal, g, max_iterations=600, step_size=1.5, goal_bias=0.2)))
        if rrt_star is not None:
            algs.append(("RRT*", lambda: rrt_star(start, goal, g, max_iterations=600, step_size=1.5, goal_bias=0.2, search_radius=3.0)))
        if prm is not None:
            algs.append(("PRM", lambda: prm(start, goal, g, num_samples=120, connection_radius=4.0)))

        for name, fn in algs:
            with self.subTest(algorithm=name):
                path = _run_with_retries(fn, attempts=5)
                self.assertTrue(path, f"{name} should find a path in open grid (with retries)")
                _assert_path_valid(self, g, start, goal, path)

    def test_algorithms_no_solution_when_goal_is_isolated(self) -> None:
        """If goal is fully surrounded by obstacles (8-neighborhood), path must be []."""
        g = GridMap(10, 10)
        start: Point = (1, 1)
        goal: Point = (8, 8)

        # Surround goal
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                x, y = goal[0] + dx, goal[1] + dy
                if 0 <= x < g.width and 0 <= y < g.height:
                    g.add_obstacle(x, y)

        self.assertTrue(g.is_free(*goal), "Goal itself should remain free")

        algs: List[Tuple[str, Callable[[], List[Point]]]] = []
        if rrt is not None:
            algs.append(("RRT", lambda: rrt(start, goal, g, max_iterations=400, step_size=1.5, goal_bias=0.3)))
        if rrt_star is not None:
            algs.append(("RRT*", lambda: rrt_star(start, goal, g, max_iterations=400, step_size=1.5, goal_bias=0.3, search_radius=3.0)))
        if prm is not None:
            algs.append(("PRM", lambda: prm(start, goal, g, num_samples=80, connection_radius=3.0)))

        for name, fn in algs:
            with self.subTest(algorithm=name):
                path = _run_with_retries(fn, attempts=3)
                self.assertEqual(path, [], f"{name} should return [] when goal is isolated")

    def test_parameter_smoke(self) -> None:
        """Smoke-test a couple parameter variations (validate if returned)."""
        g = GridMap(10, 10)
        start: Point = (1, 1)
        goal: Point = (8, 8)

        if rrt is not None:
            p1 = rrt(start, goal, g, max_iterations=200, step_size=0.75, goal_bias=0.1)
            p2 = rrt(start, goal, g, max_iterations=200, step_size=2.0, goal_bias=0.2)
            _assert_path_valid(self, g, start, goal, p1)
            _assert_path_valid(self, g, start, goal, p2)

        if prm is not None:
            p1 = prm(start, goal, g, num_samples=60, connection_radius=2.0)
            p2 = prm(start, goal, g, num_samples=160, connection_radius=4.0)
            _assert_path_valid(self, g, start, goal, p1)
            _assert_path_valid(self, g, start, goal, p2)


if __name__ == "__main__":
    unittest.main()
