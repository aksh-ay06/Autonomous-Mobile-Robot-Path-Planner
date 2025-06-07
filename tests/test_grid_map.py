"""
Tests for GridMap module.
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
        assert len(grid.static_obstacles) == 0
        
        # Test with obstacles
        obstacles = {(1, 2), (3, 4)}
        grid_with_obs = GridMap(10, 5, obstacles)
        assert grid_with_obs.static_obstacles == obstacles
    
    def test_is_free_bounds_checking(self):
        """Test bounds checking in is_free method."""
        grid = GridMap(5, 5)
        
        # Valid positions
        assert grid.is_free(0, 0) == True
        assert grid.is_free(4, 4) == True
        assert grid.is_free(2, 2) == True
        
        # Out of bounds
        assert grid.is_free(-1, 0) == False
        assert grid.is_free(0, -1) == False
        assert grid.is_free(5, 0) == False
        assert grid.is_free(0, 5) == False
    
    def test_is_free_obstacles(self):
        """Test obstacle checking in is_free method."""
        obstacles = {(1, 1), (2, 3), (4, 0)}
        grid = GridMap(5, 5, obstacles)
        
        # Free positions
        assert grid.is_free(0, 0) == True
        assert grid.is_free(3, 3) == True
        
        # Obstacle positions
        assert grid.is_free(1, 1) == False
        assert grid.is_free(2, 3) == False
        assert grid.is_free(4, 0) == False
    
    def test_neighbors_4_connected(self):
        """Test 4-connected neighbors generation."""
        grid = GridMap(5, 5)
        
        # Center position
        neighbors = grid.neighbors(2, 2)
        expected = [(2, 1), (2, 3), (1, 2), (3, 2)]
        assert set(neighbors) == set(expected)
        
        # Corner position
        neighbors = grid.neighbors(0, 0)
        expected = [(0, 1), (1, 0)]
        assert set(neighbors) == set(expected)
        
        # Edge position
        neighbors = grid.neighbors(2, 0)
        expected = [(2, 1), (1, 0), (3, 0)]
        assert set(neighbors) == set(expected)
    
    def test_neighbors_with_obstacles(self):
        """Test neighbors with obstacles."""
        obstacles = {(1, 2), (3, 2)}
        grid = GridMap(5, 5, obstacles)
        
        neighbors = grid.neighbors(2, 2)
        # Should exclude obstacles at (1, 2) and (3, 2)
        expected = [(2, 1), (2, 3)]
        assert set(neighbors) == set(expected)
    
    def test_add_remove_obstacle(self):
        """Test adding and removing obstacles."""
        grid = GridMap(5, 5)
        
        # Add obstacle
        grid.add_obstacle(2, 2)
        assert (2, 2) in grid.static_obstacles
        assert grid.is_free(2, 2) == False
        
        # Remove obstacle
        grid.remove_obstacle(2, 2)
        assert (2, 2) not in grid.static_obstacles
        assert grid.is_free(2, 2) == True
        
        # Remove non-existent obstacle (should not raise error)
        grid.remove_obstacle(1, 1)
        assert (1, 1) not in grid.static_obstacles
