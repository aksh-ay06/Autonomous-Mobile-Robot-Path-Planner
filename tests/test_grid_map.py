"""
Tests for GridMap module.

Updates:
- Uses plain `assert grid.is_free(...)` style (no `== True/False` noise)
- Adds a couple robustness checks:
  - neighbors() never returns out-of-bounds
  - neighbors() never returns obstacles
  - init stores a *copy* of obstacle set if your implementation does that (handled safely)
- Keeps expectations aligned with the repo's GridMap API:
  - is_free(x, y)
  - neighbors(x, y)  -> 4-connected free neighbors
  - add_obstacle/remove_obstacle update static_obstacles
"""

import pytest

from amr_path_planner.grid_map import GridMap


class TestGridMap:
    """Test cases for GridMap class."""

    def test_initialization(self):
        """Test GridMap initialization."""
        grid = GridMap(10, 5)
        assert grid.width == 10
        assert grid.height == 5
        assert isinstance(grid.static_obstacles, set)
        assert len(grid.static_obstacles) == 0

        obstacles = {(1, 2), (3, 4)}
        grid_with_obs = GridMap(10, 5, obstacles)

        # Some implementations may copy the set; equality is what matters.
        assert grid_with_obs.static_obstacles == obstacles

    def test_is_free_bounds_checking(self):
        """Test bounds checking in is_free method."""
        grid = GridMap(5, 5)

        # Valid positions
        assert grid.is_free(0, 0)
        assert grid.is_free(4, 4)
        assert grid.is_free(2, 2)

        # Out of bounds
        assert not grid.is_free(-1, 0)
        assert not grid.is_free(0, -1)
        assert not grid.is_free(5, 0)
        assert not grid.is_free(0, 5)

    def test_is_free_obstacles(self):
        """Test obstacle checking in is_free method."""
        obstacles = {(1, 1), (2, 3), (4, 0)}
        grid = GridMap(5, 5, obstacles)

        # Free positions
        assert grid.is_free(0, 0)
        assert grid.is_free(3, 3)

        # Obstacle positions
        assert not grid.is_free(1, 1)
        assert not grid.is_free(2, 3)
        assert not grid.is_free(4, 0)

    def test_neighbors_4_connected(self):
        """Test 4-connected neighbors generation."""
        grid = GridMap(5, 5)

        # Center position
        neighbors = grid.neighbors(2, 2)
        expected = {(2, 1), (2, 3), (1, 2), (3, 2)}
        assert set(neighbors) == expected

        # Corner position
        neighbors = grid.neighbors(0, 0)
        expected = {(0, 1), (1, 0)}
        assert set(neighbors) == expected

        # Edge position
        neighbors = grid.neighbors(2, 0)
        expected = {(2, 1), (1, 0), (3, 0)}
        assert set(neighbors) == expected

        # Neighbors should always be in-bounds and free
        for x, y in grid.neighbors(2, 2):
            assert 0 <= x < grid.width
            assert 0 <= y < grid.height
            assert grid.is_free(x, y)

    def test_neighbors_with_obstacles(self):
        """Test neighbors exclude obstacles."""
        obstacles = {(1, 2), (3, 2)}
        grid = GridMap(5, 5, obstacles)

        neighbors = grid.neighbors(2, 2)
        expected = {(2, 1), (2, 3)}
        assert set(neighbors) == expected

        # Ensure no obstacle is returned
        for n in neighbors:
            assert n not in obstacles

    def test_add_remove_obstacle(self):
        """Test adding and removing obstacles."""
        grid = GridMap(5, 5)

        # Add obstacle
        grid.add_obstacle(2, 2)
        assert (2, 2) in grid.static_obstacles
        assert not grid.is_free(2, 2)

        # Remove obstacle
        grid.remove_obstacle(2, 2)
        assert (2, 2) not in grid.static_obstacles
        assert grid.is_free(2, 2)

        # Remove non-existent obstacle (should not raise)
        grid.remove_obstacle(1, 1)
        assert (1, 1) not in grid.static_obstacles
