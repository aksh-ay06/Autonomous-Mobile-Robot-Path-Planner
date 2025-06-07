"""
Tests for search algorithms module.
"""

import pytest
from amr_path_planner.grid_map import GridMap
from amr_path_planner.search_algorithms import dijkstra, astar, manhattan_distance


class TestSearchAlgorithms:
    """Test cases for search algorithms."""
    
    def setup_method(self):
        """Set up test fixtures."""
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
        start = (0, 0)
        goal = (4, 0)
        path = dijkstra(start, goal, self.empty_grid)
        
        assert len(path) == 5  # 5 steps for 4-unit distance
        assert path[0] == start
        assert path[-1] == goal
        
        # Check path is valid (adjacent cells)
        for i in range(len(path) - 1):
            curr = path[i]
            next_pos = path[i + 1]
            assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1
    
    def test_dijkstra_with_obstacles(self):
        """Test Dijkstra with obstacles."""
        start = (0, 0)
        goal = (4, 2)
        path = dijkstra(start, goal, self.grid)
        
        assert len(path) > 0  # Should find a path
        assert path[0] == start
        assert path[-1] == goal
        
        # Check no path goes through obstacles
        for pos in path:
            assert pos not in self.obstacles
    
    def test_dijkstra_no_path(self):
        """Test Dijkstra when no path exists."""
        # Create grid where goal is completely blocked
        obstacles = {(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)}
        blocked_grid = GridMap(3, 3, obstacles)
        
        start = (0, 0)
        goal = (1, 1)  # This is an obstacle
        path = dijkstra(start, goal, blocked_grid)
        
        assert len(path) == 0
    
    def test_astar_simple_path(self):
        """Test A* on simple path."""
        start = (0, 0)
        goal = (4, 0)
        path = astar(start, goal, self.empty_grid)
        
        assert len(path) == 5  # 5 steps for 4-unit distance
        assert path[0] == start
        assert path[-1] == goal
        
        # Check path is valid (adjacent cells)
        for i in range(len(path) - 1):
            curr = path[i]
            next_pos = path[i + 1]
            assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1
    
    def test_astar_with_obstacles(self):
        """Test A* with obstacles."""
        start = (0, 0)
        goal = (4, 2)
        path = astar(start, goal, self.grid)
        
        assert len(path) > 0  # Should find a path
        assert path[0] == start
        assert path[-1] == goal
        
        # Check no path goes through obstacles
        for pos in path:
            assert pos not in self.obstacles
    
    def test_astar_no_path(self):
        """Test A* when no path exists."""
        # Create grid where goal is completely blocked
        obstacles = {(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)}
        blocked_grid = GridMap(3, 3, obstacles)
        
        start = (0, 0)
        goal = (1, 1)  # This is an obstacle
        path = astar(start, goal, blocked_grid)
        
        assert len(path) == 0
    
    def test_dijkstra_vs_astar_same_result(self):
        """Test that Dijkstra and A* find paths of same length (optimal)."""
        start = (0, 0)
        goal = (4, 4)
        
        dijkstra_path = dijkstra(start, goal, self.grid)
        astar_path = astar(start, goal, self.grid)
        
        # Both should find paths
        assert len(dijkstra_path) > 0
        assert len(astar_path) > 0
        
        # Both should find optimal paths (same length)
        assert len(dijkstra_path) == len(astar_path)
    
    def test_invalid_start_goal(self):
        """Test behavior with invalid start or goal positions."""
        # Start position is obstacle
        path = dijkstra((1, 1), (4, 4), self.grid)
        assert len(path) == 0
        
        # Goal position is obstacle
        path = dijkstra((0, 0), (1, 1), self.grid)
        assert len(path) == 0
        
        # Same tests for A*
        path = astar((1, 1), (4, 4), self.grid)
        assert len(path) == 0
        
        path = astar((0, 0), (1, 1), self.grid)
        assert len(path) == 0
    
    def test_start_equals_goal(self):
        """Test when start equals goal."""
        start = goal = (2, 2)
        
        dijkstra_path = dijkstra(start, goal, self.empty_grid)
        astar_path = astar(start, goal, self.empty_grid)
        
        assert dijkstra_path == [start]
        assert astar_path == [start]
