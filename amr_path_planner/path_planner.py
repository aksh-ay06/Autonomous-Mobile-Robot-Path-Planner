"""
Path Planner wrapper for autonomous mobile robot path planning.
Provides a unified interface for different search algorithms.
"""

from typing import Tuple, List, Callable, Optional, Union
from .grid_map import GridMap
from .search_algorithms import dijkstra, astar, manhattan_distance

# Import advanced algorithms
try:
    from .advanced_algorithms import rrt, rrt_star, prm
    ADVANCED_ALGORITHMS_AVAILABLE = True
except ImportError:
    ADVANCED_ALGORITHMS_AVAILABLE = False
    rrt = rrt_star = prm = None

# Import path smoothing
try:
    from .path_smoothing import smooth_path
    PATH_SMOOTHING_AVAILABLE = True
except ImportError:
    PATH_SMOOTHING_AVAILABLE = False
    smooth_path = None

# Import enhanced grid
try:
    from .enhanced_grid import EnhancedGridMap
    ENHANCED_GRID_AVAILABLE = True
except ImportError:
    ENHANCED_GRID_AVAILABLE = False


class PathPlanner:
    """
    Enhanced path planner that wraps search algorithms and provides unified interface.
    
    Attributes:
        algorithm (str): Algorithm name
        heuristic (Callable): Heuristic function for A* (ignored for Dijkstra)
        grid (GridMap): Grid map instance
        smoothing_enabled (bool): Whether to apply path smoothing
        smoothing_method (str): Path smoothing method
        smoothing_params (dict): Parameters for path smoothing    """
    
    def __init__(self, algorithm: str = 'astar', heuristic: Optional[Callable] = None, 
                 grid: Optional[Union[GridMap, 'EnhancedGridMap']] = None,
                 enable_smoothing: bool = False, smoothing_method: str = 'shortcut'):
        """
        Initialize PathPlanner.
        
        Args:
            algorithm: Algorithm to use ('dijkstra', 'astar', 'rrt', 'rrt_star', 'prm')
            heuristic: Heuristic function for A* (default: Manhattan distance)
            grid: GridMap or EnhancedGridMap instance (can be set later)
            enable_smoothing: Whether to enable path smoothing
            smoothing_method: Path smoothing method ('shortcut', 'adaptive', 'douglas_peucker')
        """
        valid_algorithms = ['dijkstra', 'astar']
        if ADVANCED_ALGORITHMS_AVAILABLE:
            valid_algorithms.extend(['rrt', 'rrt_star', 'prm'])
        
        if algorithm not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of: {valid_algorithms}")
        
        self.algorithm = algorithm
        self.heuristic = heuristic or manhattan_distance
        self.grid = grid
        self.smoothing_enabled = enable_smoothing and PATH_SMOOTHING_AVAILABLE
        self.smoothing_method = smoothing_method
        self.smoothing_params = {}
    
    def set_grid(self, grid: Union[GridMap, 'EnhancedGridMap']):
        """Set the grid map for path planning."""
        self.grid = grid
    
    def set_smoothing_params(self, **params):
        """Set parameters for path smoothing."""
        self.smoothing_params.update(params)
    
    def compute_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Compute path from start to goal using selected algorithm.
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List[Tuple[int, int]]: Path from start to goal, empty if no path exists
            
        Raises:
            ValueError: If grid is not set or algorithm requires unavailable modules
        """
        if self.grid is None:
            raise ValueError("Grid must be set before computing path")
        
        # Compute raw path using selected algorithm
        if self.algorithm == 'dijkstra':
            path = dijkstra(start, goal, self.grid)
        elif self.algorithm == 'astar':
            path = astar(start, goal, self.grid, self.heuristic)
        elif self.algorithm == 'rrt':
            if not ADVANCED_ALGORITHMS_AVAILABLE or rrt is None:
                raise ValueError("RRT algorithm not available. Install required dependencies.")
            max_iterations = self.smoothing_params.get('max_iterations', 2000)
            step_size = self.smoothing_params.get('step_size', 1.0)
            goal_bias = self.smoothing_params.get('goal_bias', 0.1)
            path = rrt(start, goal, self.grid, max_iterations, step_size, goal_bias)
        elif self.algorithm == 'rrt_star':
            if not ADVANCED_ALGORITHMS_AVAILABLE or rrt_star is None:
                raise ValueError("RRT* algorithm not available. Install required dependencies.")
            max_iterations = self.smoothing_params.get('max_iterations', 2000)
            step_size = self.smoothing_params.get('step_size', 1.0)
            goal_bias = self.smoothing_params.get('goal_bias', 0.1)
            search_radius = self.smoothing_params.get('search_radius', 2.0)
            path = rrt_star(start, goal, self.grid, max_iterations, step_size, goal_bias, search_radius)
        elif self.algorithm == 'prm':
            if not ADVANCED_ALGORITHMS_AVAILABLE or prm is None:
                raise ValueError("PRM algorithm not available. Install required dependencies.")
            num_samples = self.smoothing_params.get('num_samples', 500)
            connection_radius = self.smoothing_params.get('connection_radius', 2.0)
            path = prm(start, goal, self.grid, num_samples, connection_radius)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Apply path smoothing if enabled and path was found
        if path and self.smoothing_enabled and PATH_SMOOTHING_AVAILABLE and smooth_path is not None:
            try:
                path = smooth_path(path, self.grid, self.smoothing_method, **self.smoothing_params)
            except Exception as e:                print(f"Warning: Path smoothing failed: {e}")
                # Continue with unsmoothed path
        
        return path
    
    def change_algorithm(self, algorithm: str):
        """Change the search algorithm."""
        valid_algorithms = ['dijkstra', 'astar']
        if ADVANCED_ALGORITHMS_AVAILABLE:
            valid_algorithms.extend(['rrt', 'rrt_star', 'prm'])
        
        if algorithm not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of: {valid_algorithms}")
        self.algorithm = algorithm
    
    def change_heuristic(self, heuristic: Callable):
        """Change the heuristic function (for A* algorithm)."""
        self.heuristic = heuristic
