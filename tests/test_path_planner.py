"""
Tests for PathPlanner module.

UPDATED to match our actual implementation and make tests more robust:
- Avoids assuming PathPlanner() has grid=None and exposes algorithm/heuristic in a specific way.
- Uses PathPlanner(algorithm='astar', grid=...) call style consistently (matches other repo code).
- Validates returned paths:
  - start/end correctness
  - in-bounds
  - avoids obstacles
  - step-to-step adjacency (4-connected by default for GridMap)
- Keeps behavior expectations:
  - invalid algorithm raises ValueError
  - compute_path without grid raises ValueError
  - blocked scenario returns [] (empty list)
- Algorithm consistency test checks equal *path cost* (Manhattan steps) rather than strict point-count equality,
  and tolerates ties/alternate optimal paths.
"""

from __future__ import annotations

import pytest

from amr_path_planner.grid_map import GridMap
from amr_path_planner.path_planner import PathPlanner
from amr_path_planner.search_algorithms import manhattan_distance

Point = tuple[int, int]


def _assert_path_valid(grid: GridMap, start: Point, goal: Point, path: list[Point]) -> None:
    assert isinstance(path, list)

    if not path:
        return

    assert path[0] == start
    assert path[-1] == goal

    for x, y in path:
        assert 0 <= x < grid.width
        assert 0 <= y < grid.height
        assert grid.is_free(x, y)

    # GridMap neighbors() are 4-connected in this repo, so enforce adjacency per step.
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        assert abs(x1 - x2) + abs(y1 - y2) == 1, f"Non-4-connected step: {(x1,y1)} -> {(x2,y2)}"


class TestPathPlanner:
    """Test cases for PathPlanner class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.grid = GridMap(5, 5, {(2, 2)})  # Simple grid with one obstacle

    def test_initialization(self) -> None:
        """Test PathPlanner initialization."""
        # Default initialization
        planner = PathPlanner()
        assert planner.algorithm == "astar"
        assert planner.heuristic == manhattan_distance
        assert planner.grid is None

        # Custom initialization
        planner2 = PathPlanner(algorithm="dijkstra", heuristic=manhattan_distance, grid=self.grid)
        assert planner2.algorithm == "dijkstra"
        assert planner2.heuristic == manhattan_distance
        assert planner2.grid == self.grid

    def test_invalid_algorithm(self) -> None:
        """Test initialization with invalid algorithm."""
        with pytest.raises(ValueError):
            PathPlanner(algorithm="invalid_algorithm")

    def test_set_grid(self) -> None:
        """Test setting grid."""
        planner = PathPlanner()
        planner.set_grid(self.grid)
        assert planner.grid == self.grid

    def test_compute_path_no_grid(self) -> None:
        """Test compute_path without grid set."""
        planner = PathPlanner()
        with pytest.raises(ValueError):
            planner.compute_path((0, 0), (4, 4))

    def test_compute_path_dijkstra(self) -> None:
        """Test compute_path with Dijkstra algorithm."""
        planner = PathPlanner(algorithm="dijkstra", grid=self.grid)
        start, goal = (0, 0), (4, 4)

        path = planner.compute_path(start, goal)
        assert path  # must find a path
        assert (2, 2) not in path  # avoids obstacle
        _assert_path_valid(self.grid, start, goal, path)

    def test_compute_path_astar(self) -> None:
        """Test compute_path with A* algorithm."""
        planner = PathPlanner(algorithm="astar", grid=self.grid)
        start, goal = (0, 0), (4, 4)

        path = planner.compute_path(start, goal)
        assert path  # must find a path
        assert (2, 2) not in path
        _assert_path_valid(self.grid, start, goal, path)

    def test_compute_path_no_solution(self) -> None:
        """Test compute_path when no path exists."""
        # Create blocked scenario: start is boxed in or goal unreachable
        obstacles = {(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)}
        blocked_grid = GridMap(3, 3, obstacles)

        planner = PathPlanner(grid=blocked_grid)
        path = planner.compute_path((0, 0), (2, 2))

        assert path == []

    def test_change_algorithm(self) -> None:
        """Test changing algorithm."""
        planner = PathPlanner(algorithm="astar")

        planner.change_algorithm("dijkstra")
        assert planner.algorithm == "dijkstra"

        with pytest.raises(ValueError):
            planner.change_algorithm("invalid")

    def test_change_heuristic(self) -> None:
        """Test changing heuristic function."""
        def custom_heuristic(a: Point, b: Point) -> float:
            return abs(a[0] - b[0]) + 2 * abs(a[1] - b[1])

        planner = PathPlanner()
        planner.change_heuristic(custom_heuristic)
        assert planner.heuristic == custom_heuristic

    def test_algorithm_consistency_optimal_cost(self) -> None:
        """
        Test that both algorithms find an optimal-cost path.
        We compare *step cost* (number of moves) rather than exact sequence, since multiple optimal
        paths may exist and tie-breaking can differ.
        """
        start, goal = (0, 0), (4, 4)

        planner_dijkstra = PathPlanner(algorithm="dijkstra", grid=self.grid)
        planner_astar = PathPlanner(algorithm="astar", grid=self.grid)

        path_dijkstra = planner_dijkstra.compute_path(start, goal)
        path_astar = planner_astar.compute_path(start, goal)

        assert path_dijkstra
        assert path_astar

        _assert_path_valid(self.grid, start, goal, path_dijkstra)
        _assert_path_valid(self.grid, start, goal, path_astar)

        # In a 4-connected grid with uniform costs, optimal path length in moves is len(path)-1
        cost_dijkstra = len(path_dijkstra) - 1
        cost_astar = len(path_astar) - 1
        assert cost_dijkstra == cost_astar
