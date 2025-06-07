"""
Simulator for autonomous mobile robot path planning.
Provides visualization and simulation loop functionality.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Tuple, Optional, List, Union
from .grid_map import GridMap
from .robot_agent import RobotAgent
from .dynamic_obstacles import DynamicObstacleMgr


class Simulator:
    """
    Simulator that runs the main loop and handles visualization.
    Supports both single robot and multi-robot scenarios.
    
    Attributes:
        grid (GridMap): Grid map instance
        agent (RobotAgent): Single robot agent (for single robot mode)
        multi_robot_coordinator: Multi-robot coordinator (for multi-robot mode)
        obstacle_mgr (DynamicObstacleMgr): Dynamic obstacle manager
        step_delay (float): Time delay between simulation steps
        max_steps (int): Maximum number of simulation steps
        is_multi_robot (bool): Whether this is a multi-robot simulation
    """
    
    def __init__(self, grid: GridMap, 
                 agent: Optional[RobotAgent] = None,
                 multi_robot_coordinator = None,
                 obstacle_mgr: Optional[DynamicObstacleMgr] = None,
                 step_delay: float = 0.1, max_steps: int = 1000):
        """
        Initialize Simulator.
        
        Args:
            grid: GridMap instance
            agent: RobotAgent instance (for single robot mode)
            multi_robot_coordinator: MultiRobotCoordinator instance (for multi-robot mode)
            obstacle_mgr: DynamicObstacleMgr instance
            step_delay: Time delay between steps in seconds
            max_steps: Maximum number of simulation steps
        """
        self.grid = grid
        self.agent = agent
        self.multi_robot_coordinator = multi_robot_coordinator
        self.obstacle_mgr = obstacle_mgr
        self.step_delay = step_delay
        self.max_steps = max_steps
        self.current_step = 0
        
        # Determine simulation mode
        self.is_multi_robot = multi_robot_coordinator is not None
        
        if not self.is_multi_robot and agent is None:
            raise ValueError("Either agent or multi_robot_coordinator must be provided")
        
        # Visualization setup
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.im = None
        self.animation = None
    
    def run(self, visualize: bool = True, save_gif: bool = False, gif_filename: str = "simulation.gif"):
        """
        Run the simulation loop.
        
        Args:
            visualize: Whether to show real-time visualization
            save_gif: Whether to save animation as GIF
            gif_filename: Filename for saved GIF
        """
        if visualize:
            if save_gif:
                self.animation = animation.FuncAnimation(
                    self.fig, self._animate_step, frames=self.max_steps,
                    interval=self.step_delay * 1000, repeat=False, blit=False
                )
                self.animation.save(gif_filename, writer='pillow', fps=10)
                print(f"Animation saved as {gif_filename}")
                        else:
                self.animation = animation.FuncAnimation(
                    self.fig, self._animate_step, frames=self.max_steps,
                    interval=self.step_delay * 1000, repeat=False, blit=False
                )
                plt.show()
        else:
            # Run without visualization
            for step in range(self.max_steps):
                if self._simulation_step():
                    break
                if step % 100 == 0:
                    if self.is_multi_robot:
                        stats = self.multi_robot_coordinator.get_coordination_statistics()
                        print(f"Step {step}: {stats['robots_at_goal']}/{stats['total_robots']} robots at goal")
                    else:
                        print(f"Step {step}: Robot at {self.agent.position}")
    
    def _animate_step(self, frame):
        """Animation function for matplotlib."""
        if self._simulation_step():
            return []  # Stop animation if goal reached
        
        self._render()
        return []
      def _simulation_step(self) -> bool:
        """
        Execute one simulation step.
        
        Returns:
            bool: True if simulation should stop (goal reached or max steps)
        """
        # Update dynamic obstacles
        if self.obstacle_mgr:
            self.obstacle_mgr.update()
        
        # Move robot(s)
        if self.is_multi_robot:
            # Multi-robot mode
            self.multi_robot_coordinator.step_all_robots(self.obstacle_mgr)
            
            # Check if all robots reached their goals
            stats = self.multi_robot_coordinator.get_coordination_statistics()
            if stats['robots_at_goal'] == stats['total_robots'] and stats['total_robots'] > 0:
                if not hasattr(self, '_goal_reached_printed'):
                    print(f"All robots reached their goals at step {self.current_step}!")
                    self._goal_reached_printed = True
                return True
        else:
            # Single robot mode
            self.agent.step(self.obstacle_mgr)
            
            if self.agent.is_at_goal():
                if not hasattr(self, '_goal_reached_printed'):
                    print(f"Goal reached at step {self.current_step}!")
                    self._goal_reached_printed = True
                return True
        
        self.current_step += 1
        
        if self.current_step >= self.max_steps:
            print(f"Maximum steps ({self.max_steps}) reached!")
            return True
        
        return False
    
    def _render(self):
        """Render the current state of the simulation."""
        # Create visualization grid
        vis_grid = np.zeros((self.grid.height, self.grid.width, 3))
        
        # Set background (free spaces) to white
        vis_grid[:, :] = [1.0, 1.0, 1.0]
        
        # Draw static obstacles (black)
        for x, y in self.grid.static_obstacles:
            if 0 <= x < self.grid.width and 0 <= y < self.grid.height:
                vis_grid[y, x] = [0.0, 0.0, 0.0]
        
        # Draw dynamic obstacles (red)
        for x, y in self.obstacle_mgr.get_obstacle_positions():
            if 0 <= x < self.grid.width and 0 <= y < self.grid.height:
                vis_grid[y, x] = [1.0, 0.0, 0.0]
        
        # Draw path (light blue)
        remaining_path = self.agent.get_remaining_path()
        for x, y in remaining_path:
            if 0 <= x < self.grid.width and 0 <= y < self.grid.height:
                vis_grid[y, x] = [0.7, 0.9, 1.0]
        
        # Draw goal (green)
        if self.agent.goal:
            gx, gy = self.agent.goal
            if 0 <= gx < self.grid.width and 0 <= gy < self.grid.height:
                vis_grid[gy, gx] = [0.0, 1.0, 0.0]
        
        # Draw robot (blue)
        rx, ry = self.agent.position
        if 0 <= rx < self.grid.width and 0 <= ry < self.grid.height:
            vis_grid[ry, rx] = [0.0, 0.0, 1.0]
        
        # Update display
        if self.im is None:
            self.im = self.ax.imshow(vis_grid, origin='upper')
            self.ax.set_title(f"AMR Path Planning Simulation")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            
            # Add grid lines
            self.ax.set_xticks(np.arange(-0.5, self.grid.width, 1), minor=True)
            self.ax.set_yticks(np.arange(-0.5, self.grid.height, 1), minor=True)
            self.ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
            
            # Add legend
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, fc="blue", label="Robot"),
                plt.Rectangle((0, 0), 1, 1, fc="green", label="Goal"),
                plt.Rectangle((0, 0), 1, 1, fc="red", label="Dynamic Obstacle"),
                plt.Rectangle((0, 0), 1, 1, fc="black", label="Static Obstacle"),
                plt.Rectangle((0, 0), 1, 1, fc="lightblue", label="Path")
            ]
            self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
            
        else:
            self.im.set_array(vis_grid)
        
        # Update title with current step
        self.ax.set_title(f"AMR Path Planning Simulation - Step {self.current_step}")
        
        plt.tight_layout()
    
    def step_once(self):
        """Execute a single simulation step (useful for debugging)."""
        return self._simulation_step()
    
    def reset(self):
        """Reset simulation to initial state."""
        self.current_step = 0
        # Note: You may want to reset agent and obstacle positions here
    
    def get_statistics(self) -> dict:
        """Get simulation statistics."""
        return {
            "current_step": self.current_step,
            "robot_position": self.agent.position,
            "goal_position": self.agent.goal,
            "goal_reached": self.agent.is_at_goal(),
            "path_length": len(self.agent.path),
            "remaining_path_length": len(self.agent.get_remaining_path()),
            "dynamic_obstacles_count": len(self.obstacle_mgr.get_obstacle_positions())
        }
