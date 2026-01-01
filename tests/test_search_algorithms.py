"""
Tests for search_algorithms module.

UPDATED for robustness + alignment with actual GridMap API:
- Avoids assuming exact path shape beyond optimality/validity.
- Verifies:
  - endpoints
  - 4-connected adjacency between steps
  - in-bounds
  - avoids obstacles
- Uses "goal blocked" cases where goal is an obstacle (should return []).
- Allows multiple optimal paths (tie-breaking differences), so compares optimal cost not the exact sequence.
"""

from __future__ import annotations

import pytest

from amr_path_planner.grid_map import GridMap
from amr_path_planner.search_algorithms import astar, dijkstra, manhattan_distance

Point = tuple[int, int]


def _assert_path_valid(grid: GridMap, path: list[Point], start: Point, goal: Point) -> None:
    """Shared validator for returned paths."""
    assert isinstance(path, list)

    if not path:
        return

    assert path[0] == start
    assert path[-1] == goal

    for x, y in path:
        assert 0 <= x < grid.width
        assert 0 <= y < grid.height
        assert grid.is_free(x, y)

    # 4-connected adjacency
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        assert abs(x1 - x2) + abs(y1 - y2) == 1, f"Non-adjacent step: {(x1,y1)} -> {(x2,y2)}"


class TestSearchAlgorithms:
    """Test cases for search algorithms."""

    def setup_method(self) -> None:
        # Simple 5x5 grid with some obstacles
        self.obstacles = {(1, 1), (2, 1), (3, 1), (1, 3), (2, 3)}
        self.grid = GridMap(5, 5, self.obstacles)

        # Empty grid for simple tests
        self.empty_grid = GridMap(5, 5)

    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        assert manhattan_distance((0, 0), (3, 4)) == 7
        assert manhattan_distance((2, 2), (2, 2)) == 0
        assert manhattan_distance((1, 3), (4, 1)) == 5

    def test_dijkstra_simple_path(self):
        """Test Dijkstra on simple path."""
        start, goal = (0, 0), (4, 0)
        path = dijkstra(start, goal, self.empty_grid)

        assert path  # must find
        _assert_path_valid(self.empty_grid, path, start, goal)

        # Optimal move cost equals Manhattan distance in empty 4-connected grid
        assert (len(path) - 1) == manhattan_distance(start, goal)

    def test_dijkstra_with_obstacles(self):
        """Test Dijkstra with obstacles."""
        start, goal = (0, 0), (4, 2)
        path = dijkstra(start, goal, self.grid)

        assert path
        _assert_path_valid(self.grid, path, start, goal)

        for pos in path:
            assert pos not in self.obstacles

    def test_dijkstra_no_path_when_goal_blocked(self):
        """Test Dijkstra when goal is an obstacle."""
        obstacles = {(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)}
        blocked_grid = GridMap(3, 3, obstacles)

        start = (0, 0)
        goal = (1, 1)  # obstacle
        path = dijkstra(start, goal, blocked_grid)

        assert path == []

    def test_astar_simple_path(self):
        """Test A* on simple path."""
        start, goal = (0, 0), (4, 0)
        path = astar(start, goal, self.empty_grid)

        assert path
        _assert_path_valid(self.empty_grid, path, start, goal)

        assert (len(path) - 1) == manhattan_distance(start, goal)

    def test_astar_with_obstacles(self):
        """Test A* with obstacles."""
        start, goal = (0, 0), (4, 2)
        path = astar(start, goal, self.grid)

        assert path
        _assert_path_valid(self.grid, path, start, goal)

        for pos in path:
            assert pos not in self.obstacles

    def test_astar_no_path_when_goal_blocked(self):
        """Test A* when goal is an obstacle."""
        obstacles = {(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)}
        blocked_grid = GridMap(3, 3, obstacles)

        start = (0, 0)
        goal = (1, 1)  # obstacle
        path = astar(start, goal, blocked_grid)

        assert path == []

    def test_dijkstra_vs_astar_same_optimal_cost(self):
        """Test that Dijkstra and A* return paths with the same optimal cost."""
        start, goal = (0, 0), (4, 4)

        d_path = dijkstra(start, goal, self.grid)
        a_path = astar(start, goal, self.grid)

        assert d_path
        assert a_path

        _assert_path_valid(self.grid, d_path, start, goal)
        _assert_path_valid(self.grid, a_path, start, goal)

        # Compare cost (#moves). Do NOT compare exact sequences (tie-breaking differs).
        assert (len(d_path) - 1) == (len(a_path) - 1)

    def test_invalid_start_goal(self):
        """Test behavior with invalid start or goal positions (obstacles)."""
        # Start position is obstacle
        assert dijkstra((1, 1), (4, 4), self.grid) == []
        assert astar((1, 1), (4, 4), self.grid) == []

        # Goal position is obstacle
        assert dijkstra((0, 0), (1, 1), self.grid) == []
        assert astar((0, 0), (1, 1), self.grid) == []

    def test_start_equals_goal(self):
        """Test when start equals goal."""
        start = goal = (2, 2)

        d_path = dijkstra(start, goal, self.empty_grid)
        a_path = astar(start, goal, self.empty_grid)

        assert d_path == [start]
        assert a_path == [start]
