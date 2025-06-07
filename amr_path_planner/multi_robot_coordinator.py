"""
Multi-robot coordinator for autonomous mobile robot path planning.
Handles coordination between multiple robots to avoid collisions and optimize paths.
"""

from typing import List, Dict, Tuple, Set, Optional
import time
from .grid_map import GridMap
from .robot_agent import RobotAgent
from .path_planner import PathPlanner
from .dynamic_obstacles import DynamicObstacleMgr


class MultiRobotCoordinator:
    """
    Coordinates multiple robots to avoid collisions and optimize overall performance.
    
    Attributes:
        robots (List[RobotAgent]): List of robot agents
        grid (GridMap): Shared grid map
        coordination_mode (str): Coordination strategy ('priority', 'cooperative', 'centralized')
        robot_goals (Dict[int, Tuple[int, int]]): Goals for each robot
        robot_priorities (Dict[int, int]): Priority levels for each robot
    """
    
    def __init__(self, grid: GridMap, coordination_mode: str = 'priority'):
        """
        Initialize MultiRobotCoordinator.
        
        Args:
            grid: Shared grid map for all robots
            coordination_mode: Coordination strategy
                - 'priority': Higher priority robots plan first
                - 'cooperative': Robots negotiate paths
                - 'centralized': Central planner optimizes all paths
        """
        if coordination_mode not in ['priority', 'cooperative', 'centralized']:
            raise ValueError("coordination_mode must be 'priority', 'cooperative', or 'centralized'")
        
        self.grid = grid
        self.coordination_mode = coordination_mode
        self.robots: List[RobotAgent] = []
        self.robot_goals: Dict[int, Tuple[int, int]] = {}
        self.robot_priorities: Dict[int, int] = {}
        self.reserved_positions: Dict[Tuple[int, int], int] = {}  # position -> robot_id
        self.future_positions: Dict[int, List[Tuple[int, int]]] = {}  # robot_id -> future path
        
    def add_robot(self, robot: RobotAgent, priority: int = 0) -> int:
        """
        Add a robot to the coordination system.
        
        Args:
            robot: RobotAgent to add
            priority: Priority level (higher = more priority)
            
        Returns:
            int: Robot ID assigned to this robot
        """
        robot_id = len(self.robots)
        self.robots.append(robot)
        self.robot_priorities[robot_id] = priority
        self.future_positions[robot_id] = []
        return robot_id
    
    def set_robot_goal(self, robot_id: int, goal: Tuple[int, int]):
        """Set goal for a specific robot."""
        if robot_id >= len(self.robots):
            raise ValueError(f"Robot ID {robot_id} does not exist")
        self.robot_goals[robot_id] = goal
    
    def get_robot_positions(self) -> Dict[int, Tuple[int, int]]:
        """Get current positions of all robots."""
        return {i: robot.position for i, robot in enumerate(self.robots)}
    
    def get_occupied_positions(self, exclude_robot: int = -1) -> Set[Tuple[int, int]]:
        """
        Get positions currently occupied by robots.
        
        Args:
            exclude_robot: Robot ID to exclude from occupied positions
            
        Returns:
            Set of occupied positions
        """
        occupied = set()
        for i, robot in enumerate(self.robots):
            if i != exclude_robot:
                occupied.add(robot.position)
        return occupied
    
    def get_future_occupied_positions(self, exclude_robot: int = -1, 
                                    steps_ahead: int = 5) -> Set[Tuple[int, int]]:
        """
        Get positions that will be occupied by robots in the near future.
        
        Args:
            exclude_robot: Robot ID to exclude
            steps_ahead: Number of steps to look ahead
            
        Returns:
            Set of future occupied positions
        """
        future_occupied = set()
        for robot_id, future_path in self.future_positions.items():
            if robot_id != exclude_robot and future_path:
                # Add next few positions from each robot's path
                for i in range(min(steps_ahead, len(future_path))):
                    future_occupied.add(future_path[i])
        return future_occupied
    
    def plan_coordinated_paths(self):
        """Plan paths for all robots with coordination."""
        if self.coordination_mode == 'priority':
            self._plan_priority_based()
        elif self.coordination_mode == 'cooperative':
            self._plan_cooperative()
        elif self.coordination_mode == 'centralized':
            self._plan_centralized()
    
    def _plan_priority_based(self):
        """Plan paths using priority-based coordination."""
        # Sort robots by priority (higher priority first)
        sorted_robots = sorted(enumerate(self.robots), 
                             key=lambda x: self.robot_priorities[x[0]], reverse=True)
        
        self.reserved_positions.clear()
        
        for robot_id, robot in sorted_robots:
            if robot_id in self.robot_goals:
                goal = self.robot_goals[robot_id]
                
                # Create temporary obstacles from other robots' positions and reserved paths
                temp_obstacles = self.grid.static_obstacles.copy()
                
                # Add current positions of other robots
                occupied = self.get_occupied_positions(exclude_robot=robot_id)
                temp_obstacles.update(occupied)
                
                # Add reserved positions from higher priority robots
                temp_obstacles.update(self.reserved_positions.keys())
                
                # Create temporary grid with additional obstacles
                temp_grid = GridMap(self.grid.width, self.grid.height, temp_obstacles)
                robot.planner.set_grid(temp_grid)
                
                # Plan path
                robot.plan_to(goal)
                
                # Reserve positions along this robot's path
                if robot.has_path():
                    path = robot.get_remaining_path()
                    self.future_positions[robot_id] = path
                    for pos in path[1:]:  # Don't reserve current position
                        self.reserved_positions[pos] = robot_id
                
                # Restore original grid
                robot.planner.set_grid(self.grid)
    
    def _plan_cooperative(self):
        """Plan paths using cooperative coordination with negotiation."""
        max_iterations = 10
        
        for iteration in range(max_iterations):
            conflicts = self._detect_path_conflicts()
            if not conflicts:
                break  # No conflicts, planning complete
            
            # Resolve conflicts by replanning lower priority robots
            for conflict in conflicts:
                robot1_id, robot2_id, position = conflict
                
                # Lower priority robot replans
                if self.robot_priorities[robot1_id] < self.robot_priorities[robot2_id]:
                    lower_priority_robot = robot1_id
                else:
                    lower_priority_robot = robot2_id
                
                self._replan_robot_avoiding_others(lower_priority_robot)
    
    def _plan_centralized(self):
        """Plan paths using centralized coordination (simplified version)."""
        # This is a simplified centralized planner
        # In practice, this would use more sophisticated algorithms like CBS (Conflict-Based Search)
        
        # For now, implement as priority-based with dynamic priority adjustment
        self._plan_priority_based()
        
        # Check for deadlocks and adjust
        deadlock_count = 0
        max_deadlock_iterations = 5
        
        while self._has_deadlock() and deadlock_count < max_deadlock_iterations:
            # Randomly adjust priorities and replan
            import random
            robot_ids = list(range(len(self.robots)))
            random.shuffle(robot_ids)
            
            for i, robot_id in enumerate(robot_ids):
                self.robot_priorities[robot_id] = len(robot_ids) - i
            
            self._plan_priority_based()
            deadlock_count += 1
    
    def _detect_path_conflicts(self) -> List[Tuple[int, int, Tuple[int, int]]]:
        """
        Detect conflicts between robot paths.
        
        Returns:
            List of conflicts: (robot1_id, robot2_id, conflict_position)
        """
        conflicts = []
        
        for i in range(len(self.robots)):
            for j in range(i + 1, len(self.robots)):
                path1 = self.future_positions.get(i, [])
                path2 = self.future_positions.get(j, [])
                
                # Check for position conflicts at same time steps
                min_len = min(len(path1), len(path2))
                for step in range(min_len):
                    if path1[step] == path2[step]:
                        conflicts.append((i, j, path1[step]))
                
                # Check for edge conflicts (robots swapping positions)
                for step in range(min_len - 1):
                    if (path1[step] == path2[step + 1] and 
                        path1[step + 1] == path2[step]):
                        conflicts.append((i, j, path1[step]))
        
        return conflicts
    
    def _replan_robot_avoiding_others(self, robot_id: int):
        """Replan path for a specific robot avoiding other robots."""
        if robot_id not in self.robot_goals:
            return
        
        robot = self.robots[robot_id]
        goal = self.robot_goals[robot_id]
        
        # Create obstacles from other robots
        temp_obstacles = self.grid.static_obstacles.copy()
        
        # Add positions from other robots' paths
        for other_id, other_path in self.future_positions.items():
            if other_id != robot_id:
                temp_obstacles.update(other_path)
        
        # Create temporary grid and replan
        temp_grid = GridMap(self.grid.width, self.grid.height, temp_obstacles)
        robot.planner.set_grid(temp_grid)
        robot.plan_to(goal)
        
        # Update future positions
        if robot.has_path():
            self.future_positions[robot_id] = robot.get_remaining_path()
        
        # Restore original grid
        robot.planner.set_grid(self.grid)
    
    def _has_deadlock(self) -> bool:
        """Check if robots are in a deadlock situation."""
        # Simple deadlock detection: check if any robot hasn't moved for several steps
        for robot_id, robot in enumerate(self.robots):
            if (robot_id in self.robot_goals and 
                robot.goal and 
                not robot.has_path() and 
                not robot.is_at_goal()):
                return True
        return False
    
    def step_all_robots(self, obstacle_mgr: Optional[DynamicObstacleMgr] = None):
        """
        Execute one simulation step for all robots with coordination.
        
        Args:
            obstacle_mgr: Dynamic obstacle manager
        """
        # Update future positions
        for robot_id, robot in enumerate(self.robots):
            if robot.has_path():
                remaining_path = robot.get_remaining_path()
                self.future_positions[robot_id] = remaining_path
        
        # Check for immediate collisions and replan if necessary
        if self.coordination_mode in ['cooperative', 'centralized']:
            self._resolve_immediate_conflicts()
        
        # Move robots in priority order to avoid conflicts
        sorted_robots = sorted(enumerate(self.robots), 
                             key=lambda x: self.robot_priorities[x[0]], reverse=True)
        
        for robot_id, robot in sorted_robots:
            # Create dynamic obstacle manager that includes other robots
            combined_obstacles = self._create_combined_obstacle_manager(
                obstacle_mgr, exclude_robot=robot_id)
            
            # Step the robot
            robot.step(combined_obstacles)
    
    def _resolve_immediate_conflicts(self):
        """Resolve immediate conflicts between robots."""
        # Find robots that are about to collide
        next_positions = {}
        for robot_id, robot in enumerate(self.robots):
            if robot.has_path():
                remaining_path = robot.get_remaining_path()
                if len(remaining_path) > 1:
                    next_positions[robot_id] = remaining_path[1]
        
        # Check for position conflicts
        position_conflicts = {}
        for robot_id, next_pos in next_positions.items():
            if next_pos in position_conflicts:
                # Conflict detected - lower priority robot waits
                other_robot_id = position_conflicts[next_pos]
                if self.robot_priorities[robot_id] < self.robot_priorities[other_robot_id]:
                    # This robot waits (clear its immediate path)
                    self.robots[robot_id].path = [self.robots[robot_id].position]
                else:
                    # Other robot waits
                    self.robots[other_robot_id].path = [self.robots[other_robot_id].position]
                    position_conflicts[next_pos] = robot_id
            else:
                position_conflicts[next_pos] = robot_id
    
    def _create_combined_obstacle_manager(self, obstacle_mgr: Optional[DynamicObstacleMgr], 
                                        exclude_robot: int) -> 'CombinedObstacleManager':
        """Create an obstacle manager that includes other robots as obstacles."""
        return CombinedObstacleManager(obstacle_mgr, self, exclude_robot)
    
    def get_coordination_statistics(self) -> Dict:
        """Get statistics about the coordination system."""
        total_robots = len(self.robots)
        robots_at_goal = sum(1 for robot in self.robots if robot.is_at_goal())
        robots_with_paths = sum(1 for robot in self.robots if robot.has_path())
        
        return {
            "total_robots": total_robots,
            "robots_at_goal": robots_at_goal,
            "robots_with_paths": robots_with_paths,
            "coordination_mode": self.coordination_mode,
            "completion_rate": robots_at_goal / total_robots if total_robots > 0 else 0,
        }


class CombinedObstacleManager:
    """
    Obstacle manager that combines dynamic obstacles with other robots.
    """
    
    def __init__(self, obstacle_mgr: Optional[DynamicObstacleMgr], 
                 coordinator: MultiRobotCoordinator, exclude_robot: int):
        self.obstacle_mgr = obstacle_mgr
        self.coordinator = coordinator
        self.exclude_robot = exclude_robot
    
    def is_collision(self, x: int, y: int) -> bool:
        """Check collision with dynamic obstacles and other robots."""
        # Check dynamic obstacles
        if self.obstacle_mgr and self.obstacle_mgr.is_collision(x, y):
            return True
        
        # Check collision with other robots
        for robot_id, robot in enumerate(self.coordinator.robots):
            if robot_id != self.exclude_robot and robot.position == (x, y):
                return True
        
        return False
    
    def update(self):
        """Update dynamic obstacles (other robots are updated by coordinator)."""
        if self.obstacle_mgr:
            self.obstacle_mgr.update()
    
    def get_obstacle_positions(self) -> List[Tuple[int, int]]:
        """Get all obstacle positions including other robots."""
        positions = []
        
        # Add dynamic obstacles
        if self.obstacle_mgr:
            positions.extend(self.obstacle_mgr.get_obstacle_positions())
        
        # Add other robot positions
        for robot_id, robot in enumerate(self.coordinator.robots):
            if robot_id != self.exclude_robot:
                positions.append(robot.position)
        
        return positions
