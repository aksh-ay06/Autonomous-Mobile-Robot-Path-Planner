"""
Tests for PathPlanner module.
"""

import pytest
from amr_path_planner.grid_map import GridMap
from amr_path_planner.path_planner import PathPlanner
from amr_path_planner.search_algorithms import manhattan_distance


class TestPathPlanner:
    """Test cases for PathPlanner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.grid = GridMap(5, 5, {(2, 2)})  # Simple grid with one obstacle
    
    def test_initialization(self):
        """Test PathPlanner initialization."""
        # Default initialization
        planner = PathPlanner()
        assert planner.algorithm == 'astar'
        assert planner.heuristic == manhattan_distance
        assert planner.grid is None
        
        # Custom initialization
        planner = PathPlanner('dijkstra', manhattan_distance, self.grid)
        assert planner.algorithm == 'dijkstra'
        assert planner.heuristic == manhattan_distance
        assert planner.grid == self.grid
    
    def test_invalid_algorithm(self):
        """Test initialization with invalid algorithm."""
        with pytest.raises(ValueError):
            PathPlanner('invalid_algorithm')
    
    def test_set_grid(self):
        """Test setting grid."""
        planner = PathPlanner()
        planner.set_grid(self.grid)
        assert planner.grid == self.grid
    
    def test_compute_path_no_grid(self):
        """Test compute_path without grid set."""
        planner = PathPlanner()
        with pytest.raises(ValueError):
            planner.compute_path((0, 0), (4, 4))
    
    def test_compute_path_dijkstra(self):
        """Test compute_path with Dijkstra algorithm."""
        planner = PathPlanner('dijkstra', grid=self.grid)
        path = planner.compute_path((0, 0), (4, 4))
        
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
        assert (2, 2) not in path  # Should avoid obstacle
    
    def test_compute_path_astar(self):
        """Test compute_path with A* algorithm."""
        planner = PathPlanner('astar', grid=self.grid)
        path = planner.compute_path((0, 0), (4, 4))
        
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
        assert (2, 2) not in path  # Should avoid obstacle
    
    def test_compute_path_no_solution(self):
        """Test compute_path when no path exists."""
        # Create completely blocked scenario
        obstacles = {(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)}
        blocked_grid = GridMap(3, 3, obstacles)
        
        planner = PathPlanner(grid=blocked_grid)
        path = planner.compute_path((0, 0), (2, 2))
        
        assert len(path) == 0
    
    def test_change_algorithm(self):
        """Test changing algorithm."""
        planner = PathPlanner('astar')
        
        planner.change_algorithm('dijkstra')
        assert planner.algorithm == 'dijkstra'
        
        with pytest.raises(ValueError):
            planner.change_algorithm('invalid')
    
    def test_change_heuristic(self):
        """Test changing heuristic function."""
        def custom_heuristic(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) * 2
        
        planner = PathPlanner()
        planner.change_heuristic(custom_heuristic)
        assert planner.heuristic == custom_heuristic
    
    def test_algorithm_consistency(self):
        """Test that both algorithms find valid paths."""
        planner_dijkstra = PathPlanner('dijkstra', grid=self.grid)
        planner_astar = PathPlanner('astar', grid=self.grid)
        
        start, goal = (0, 0), (4, 4)
        
        path_dijkstra = planner_dijkstra.compute_path(start, goal)
        path_astar = planner_astar.compute_path(start, goal)
        
        # Both should find paths
        assert len(path_dijkstra) > 0
        assert len(path_astar) > 0
        
        # Both should start and end at correct positions
        assert path_dijkstra[0] == start and path_dijkstra[-1] == goal
        assert path_astar[0] == start and path_astar[-1] == goal
        
        # Both should find optimal paths (same length)
        assert len(path_dijkstra) == len(path_astar)
