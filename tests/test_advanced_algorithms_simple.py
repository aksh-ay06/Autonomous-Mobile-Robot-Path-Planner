"""
Simplified test suite for advanced algorithms that matches our actual implementation.

Updates in this version:
- Removes noisy prints and uses unittest skipping properly.
- Treats "probabilistic planners may fail sometimes" correctly:
  - Uses multiple attempts for success-required tests.
  - Uses deterministic "no-solution" checks with bounded attempts.
- Uses correct advanced_algorithms API expectations:
  - Node is optional (only tested if present).
  - rrt/rrt_star/prm are function-based planners returning List[(x,y)] or [].
- Adds stronger validity checks:
  - start/goal are free.
  - every point is in-bounds and free.
  - consecutive edges are collision-free (line-of-sight check).
- Keeps tests fast (small grids, bounded attempts/iterations).
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


# ---- Optional imports (advanced algorithms may not be installed in minimal environments)
try:
    from amr_path_planner.advanced_algorithms import Node, rrt, rrt_star, prm  # type: ignore

    ADVANCED_ALGORITHMS_AVAILABLE = True
except Exception:
    ADVANCED_ALGORITHMS_AVAILABLE = False
    Node = None  # type: ignore
    rrt = None  # type: ignore
    rrt_star = None  # type: ignore
    prm = None  # type: ignore


def _in_bounds(grid: GridMap, p: Point) -> bool:
    x, y = p
    return 0 <= x < grid.width and 0 <= y < grid.height


def _distance(a: Point, b: Point) -> float:
    return float(math.hypot(b[0] - a[0], b[1] - a[1]))


def _bresenham_line(a: Point, b: Point) -> List[Point]:
    """Discrete line between a and b (inclusive) for collision checking."""
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    points: List[Point] = []
    while True:
        points.append((x0, y0))
        if (x0, y0) == (x1, y1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def _edge_is_free(grid: GridMap, a: Point, b: Point) -> bool:
    """Conservative LOS check: all cells along discrete line must be free."""
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

    # Check each edge is collision-free (important for sampling planners)
    for i in range(len(path) - 1):
        testcase.assertTrue(_edge_is_free(grid, path[i], path[i + 1]), f"Edge crosses obstacle: {path[i]}->{path[i+1]}")


def _run_with_retries(fn: Callable[[], List[Point]], attempts: int) -> List[Point]:
    """Run a probabilistic planner multiple times; return first non-empty path."""
    for _ in range(attempts):
        p = fn()
        if p:
            return p
    return []


@unittest.skipUnless(ADVANCED_ALGORITHMS_AVAILABLE, "Advanced algorithms not available")
class TestAdvancedAlgorithms(unittest.TestCase):
    """Test advanced path planning algorithms."""

    def setUp(self) -> None:
        self.grid = GridMap(20, 20)
        # A horizontal wall with a gap so problem is solvable but non-trivial
        for i in range(5, 15):
            if i != 10:  # gap at (10, 10)
                self.grid.add_obstacle(i, 10)

        self.start: Point = (1, 1)
        self.goal: Point = (18, 18)

        self.assertTrue(self.grid.is_free(*self.start))
        self.assertTrue(self.grid.is_free(*self.goal))

    # ----------------------------
    # Node (optional)
    # ----------------------------

    def test_node_class(self) -> None:
        """Test the Node class if present."""
        if Node is None:
            self.skipTest("Node class not available in this build")

        node = Node((5, 5))
        self.assertEqual(node.position, (5, 5))
        self.assertIsNone(getattr(node, "parent", None))
        self.assertEqual(getattr(node, "cost", 0.0), 0.0)

        parent = Node((3, 3))
        child = Node((5, 5), parent=parent)
        self.assertEqual(child.parent, parent)

        # Some implementations track children; if present, test it.
        if hasattr(parent, "add_child") and hasattr(parent, "children"):
            parent.add_child(child)
            self.assertIn(child, parent.children)
            self.assertEqual(child.parent, parent)

    # ----------------------------
    # Core planners
    # ----------------------------

    def test_rrt_algorithm(self) -> None:
        """RRT may fail sometimes; accept [] but validate if found."""
        self.assertIsNotNone(rrt)
        path = rrt(self.start, self.goal, self.grid, max_iterations=1000, step_size=1.0, goal_bias=0.1)
        _assert_path_valid(self, self.grid, self.start, self.goal, path)

    def test_rrt_star_algorithm(self) -> None:
        """RRT* may fail sometimes; accept [] but validate if found."""
        self.assertIsNotNone(rrt_star)
        path = rrt_star(
            self.start,
            self.goal,
            self.grid,
            max_iterations=1000,
            step_size=1.0,
            goal_bias=0.1,
            search_radius=3.0,
        )
        _assert_path_valid(self, self.grid, self.start, self.goal, path)

    def test_prm_algorithm(self) -> None:
        """PRM may fail sometimes; accept [] but validate if found."""
        self.assertIsNotNone(prm)
        path = prm(self.start, self.goal, self.grid, num_samples=250, connection_radius=3.0)
        _assert_path_valid(self, self.grid, self.start, self.goal, path)

    # ----------------------------
    # Scenarios with expected outcome
    # ----------------------------

    def test_algorithms_find_path_in_simple_open_grid(self) -> None:
        """In an obstacle-free grid, all algorithms should find a path (with retries)."""
        simple_grid = GridMap(10, 10)
        start: Point = (1, 1)
        goal: Point = (8, 8)

        self.assertTrue(simple_grid.is_free(*start))
        self.assertTrue(simple_grid.is_free(*goal))

        algs: List[Tuple[str, Callable[[], List[Point]]]] = []

        if rrt is not None:
            algs.append(("RRT", lambda: rrt(start, goal, simple_grid, max_iterations=600, step_size=1.5, goal_bias=0.2)))
        if rrt_star is not None:
            algs.append(("RRT*", lambda: rrt_star(start, goal, simple_grid, max_iterations=600, step_size=1.5, goal_bias=0.2)))
        if prm is not None:
            algs.append(("PRM", lambda: prm(start, goal, simple_grid, num_samples=120, connection_radius=4.0)))

        for name, fn in algs:
            with self.subTest(algorithm=name):
                path = _run_with_retries(fn, attempts=5)
                self.assertTrue(path, f"{name} should find a path in an open grid (with retries)")
                _assert_path_valid(self, simple_grid, start, goal, path)

    def test_algorithms_no_solution_when_goal_is_isolated(self) -> None:
        """If the goal cell is fully surrounded (8-neighborhood), no collision-free entry exists."""
        blocked_grid = GridMap(10, 10)
        goal: Point = (8, 8)
        start: Point = (1, 1)

        # Surround goal (8-connected ring)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                x, y = goal[0] + dx, goal[1] + dy
                if 0 <= x < blocked_grid.width and 0 <= y < blocked_grid.height:
                    blocked_grid.add_obstacle(x, y)

        self.assertTrue(blocked_grid.is_free(*goal), "We do not block the goal itself")
        self.assertTrue(blocked_grid.is_free(*start))

        algs: List[Tuple[str, Callable[[], List[Point]]]] = []
        if rrt is not None:
            algs.append(("RRT", lambda: rrt(start, goal, blocked_grid, max_iterations=400, step_size=1.5, goal_bias=0.3)))
        if rrt_star is not None:
            algs.append(("RRT*", lambda: rrt_star(start, goal, blocked_grid, max_iterations=400, step_size=1.5, goal_bias=0.3)))
        if prm is not None:
            algs.append(("PRM", lambda: prm(start, goal, blocked_grid, num_samples=80, connection_radius=3.0)))

        for name, fn in algs:
            with self.subTest(algorithm=name):
                # Run a couple attempts to avoid "lucky" invalid behavior
                path = _run_with_retries(fn, attempts=3)
                self.assertEqual(path, [], f"{name} should return [] when goal is isolated")

    # ----------------------------
    # Parameter smoke tests
    # ----------------------------

    def test_algorithm_parameters_smoke(self) -> None:
        """Smoke-test parameter variations (valid if returned)."""
        g = GridMap(10, 10)
        start: Point = (1, 1)
        goal: Point = (8, 8)

        if rrt is not None:
            p1 = rrt(start, goal, g, max_iterations=200, step_size=0.75, goal_bias=0.1)
            p2 = rrt(start, goal, g, max_iterations=200, step_size=2.0, goal_bias=0.2)
            for p in (p1, p2):
                _assert_path_valid(self, g, start, goal, p)

        if prm is not None:
            p1 = prm(start, goal, g, num_samples=60, connection_radius=2.0)
            p2 = prm(start, goal, g, num_samples=160, connection_radius=4.0)
            for p in (p1, p2):
                _assert_path_valid(self, g, start, goal, p)


if __name__ == "__main__":
    unittest.main()
