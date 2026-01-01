# AMR Path Planner
"""Autonomous Mobile Robot Path Planner Package"""

from __future__ import annotations

__version__ = "1.2.0"
__author__ = "Akshay Patel"

# Core always-available modules
from .grid_map import GridMap
from .search_algorithms import (
    dijkstra,
    astar,
    manhattan_distance,
    euclidean_distance,
    octile_distance,
    chebyshev_distance,
)
from .path_planner import PathPlanner
from .dynamic_obstacles import DynamicObstacleMgr
from .robot_agent import RobotAgent

__all__ = [
    "GridMap",
    "dijkstra",
    "astar",
    "PathPlanner",
    "DynamicObstacleMgr",
    "RobotAgent",
    # heuristics
    "manhattan_distance",
    "euclidean_distance",
    "octile_distance",
    "chebyshev_distance",
]

# Optional imports with graceful fallback

try:
    from .simulator import Simulator
    __all__.append("Simulator")
except ImportError:
    pass

try:
    from .multi_robot_coordinator import MultiRobotCoordinator
    __all__.append("MultiRobotCoordinator")
except ImportError:
    pass

try:
    from .advanced_algorithms import rrt, rrt_star, prm
    __all__.extend(["rrt", "rrt_star", "prm"])
except ImportError:
    pass

try:
    from .path_smoothing import (
        smooth_path,
        shortcut_smoothing,
        bezier_smoothing,
        spline_smoothing,
        adaptive_smoothing,
        douglas_peucker_smoothing,
    )
    __all__.extend(
        [
            "smooth_path",
            "shortcut_smoothing",
            "bezier_smoothing",
            "spline_smoothing",
            "adaptive_smoothing",
            "douglas_peucker_smoothing",
        ]
    )
except ImportError:
    pass

try:
    from .enhanced_grid import EnhancedGridMap, MovementType, create_enhanced_grid
    __all__.extend(["EnhancedGridMap", "MovementType", "create_enhanced_grid"])
except ImportError:
    pass
