"""
Robot Agent for autonomous mobile robot path planning.
Manages robot position, path following, and replanning behavior.
"""

from typing import Tuple, List, Optional
from .grid_map import GridMap
from .path_planner import PathPlanner
from .dynamic_obstacles import DynamicObstacleMgr


class RobotAgent:
    """
    Robot agent that follows paths and replans when blocked.
    
    Attributes:
        position (Tuple[int, int]): Current robot position
        path (List[Tuple[int, int]]): Current planned path
        planner (PathPlanner): Path planner instance
        goal (Optional[Tuple[int, int]]): Current goal position
        path_index (int): Current index in the path
    """
    
    def __init__(self, start_position: Tuple[int, int], planner: PathPlanner):
        """
        Initialize RobotAgent.
        
        Args:
            start_position: Initial robot position (x, y)
            planner: PathPlanner instance
        """
        self.position = start_position
        self.path = []
        self.planner = planner
        self.goal = None
        self.path_index = 0
        self.replanning_needed = False
    
    def plan_to(self, goal: Tuple[int, int]):
        """
        Plan a path to the specified goal.
        
        Args:
            goal: Goal position (x, y)
        """
        self.goal = goal
        self.path = self.planner.compute_path(self.position, goal)
        self.path_index = 0
        self.replanning_needed = False
        
        if not self.path:
            print(f"Warning: No path found from {self.position} to {goal}")
    
    def step(self, obstacle_mgr: Optional[DynamicObstacleMgr] = None):
        """
        Move one step along the current path or trigger replanning if blocked.
        
        Args:
            obstacle_mgr: Dynamic obstacle manager for collision checking
        """
        if not self.path or self.path_index >= len(self.path):
            return  # No path or reached end
        
        # Check if we've reached the goal
        if self.position == self.goal:
            return
        
        # Get next position in path
        if self.path_index + 1 < len(self.path):
            next_position = self.path[self.path_index + 1]
        else:
            return  # Already at end of path
        
        # Check if next position is blocked
        is_blocked = False
        
        # Check against static obstacles
        if not self.planner.grid.is_free(next_position[0], next_position[1]):
            is_blocked = True
        
        # Check against dynamic obstacles
        if obstacle_mgr and obstacle_mgr.is_collision(next_position[0], next_position[1]):
            is_blocked = True
        
        if is_blocked:
            # Trigger replanning
            self.replanning_needed = True
            if self.goal:
                print(f"Path blocked at {next_position}, replanning from {self.position} to {self.goal}")
                self.plan_to(self.goal)
        else:
            # Move to next position
            self.position = next_position
            self.path_index += 1
            self.replanning_needed = False
    
    def is_at_goal(self) -> bool:
        """Check if robot has reached its goal."""
        return self.goal is not None and self.position == self.goal
    
    def has_path(self) -> bool:
        """Check if robot has a valid path."""
        return len(self.path) > 0
    
    def get_remaining_path(self) -> List[Tuple[int, int]]:
        """Get the remaining path from current position."""
        if not self.path or self.path_index >= len(self.path):
            return []
        return self.path[self.path_index:]
    
    def set_position(self, position: Tuple[int, int]):
        """
        Set robot position (useful for initialization or teleportation).
        
        Args:
            position: New position (x, y)
        """
        self.position = position
        self.path = []
        self.path_index = 0
        self.replanning_needed = False
    
    def clear_path(self):
        """Clear current path and goal."""
        self.path = []
        self.goal = None
        self.path_index = 0
        self.replanning_needed = False
    
    def force_replan(self):
        """Force replanning to current goal."""
        if self.goal:
            self.plan_to(self.goal)
