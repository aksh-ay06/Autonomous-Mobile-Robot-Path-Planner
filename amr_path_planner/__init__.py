# AMR Path Planner
"""Autonomous Mobile Robot Path Planner Package"""

__version__ = "1.1.0"
__author__ = "Akshay Patel"

from .grid_map import GridMap
from .search_algorithms import dijkstra, astar
from .path_planner import PathPlanner
from .dynamic_obstacles import DynamicObstacleMgr
from .robot_agent import RobotAgent
from .simulator import Simulator 

# Define __all__ with core modules
__all__ = [
    'GridMap',
    'dijkstra',
    'astar', 
    'PathPlanner',
    'DynamicObstacleMgr',
    'RobotAgent',
    'Simulator'
]

# Optional imports with graceful fallback
try:
    from .multi_robot_coordinator import MultiRobotCoordinator
    __all__.append('MultiRobotCoordinator')
except ImportError:
    pass

try:
    from .advanced_algorithms import rrt, rrt_star, prm
    __all__.extend(['rrt', 'rrt_star', 'prm'])
except ImportError:
    pass

try:
    from .path_smoothing import smooth_path, shortcut_smoothing, bezier_smoothing
    __all__.extend(['smooth_path', 'shortcut_smoothing', 'bezier_smoothing'])
except ImportError:
    pass

try:
    from .enhanced_grid import EnhancedGridMap, create_enhanced_grid
    __all__.extend(['EnhancedGridMap', 'create_enhanced_grid'])
except ImportError:
    pass
