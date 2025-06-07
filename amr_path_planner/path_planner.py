"""
Path Planner wrapper for autonomous mobile robot path planning.
Provides a unified interface for different search algorithms.
"""

from typing import Tuple, List, Callable, Optional
from .grid_map import GridMap
from .search_algorithms import dijkstra, astar, manhattan_distance


class PathPlanner:
    """
    Path planner that wraps search algorithms and provides unified interface.
    
    Attributes:
        algorithm (str): Algorithm name ('dijkstra' or 'astar')
        heuristic (Callable): Heuristic function for A* (ignored for Dijkstra)
        grid (GridMap): Grid map instance
    """
    
    def __init__(self, algorithm: str = 'astar', heuristic: Optional[Callable] = None, grid: Optional[GridMap] = None):
        """
        Initialize PathPlanner.
        
        Args:
            algorithm: Algorithm to use ('dijkstra' or 'astar')
            heuristic: Heuristic function for A* (default: Manhattan distance)
            grid: GridMap instance (can be set later)
        """
        if algorithm not in ['dijkstra', 'astar']:
            raise ValueError("Algorithm must be 'dijkstra' or 'astar'")
        
        self.algorithm = algorithm
        self.heuristic = heuristic or manhattan_distance
        self.grid = grid
    
    def set_grid(self, grid: GridMap):
        """Set the grid map for path planning."""
        self.grid = grid
    
    def compute_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Compute path from start to goal using selected algorithm.
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List[Tuple[int, int]]: Path from start to goal, empty if no path exists
            
        Raises:
            ValueError: If grid is not set
        """
        if self.grid is None:
            raise ValueError("Grid must be set before computing path")
        
        if self.algorithm == 'dijkstra':
            return dijkstra(start, goal, self.grid)
        elif self.algorithm == 'astar':
            return astar(start, goal, self.grid, self.heuristic)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def change_algorithm(self, algorithm: str):
        """Change the search algorithm."""
        if algorithm not in ['dijkstra', 'astar']:
            raise ValueError("Algorithm must be 'dijkstra' or 'astar'")
        self.algorithm = algorithm
    
    def change_heuristic(self, heuristic: Callable):
        """Change the heuristic function (for A* algorithm)."""
        self.heuristic = heuristic
