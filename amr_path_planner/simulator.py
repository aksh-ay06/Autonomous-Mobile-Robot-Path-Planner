"""
Simulator for autonomous mobile robot path planning.
Provides visualization and simulation loop functionality.

Supports:
- single robot (RobotAgent)
- multi robot (MultiRobotCoordinator-like)
- optional dynamic obstacles
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, TypeAlias

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .grid_map import GridMap
from .robot_agent import RobotAgent
from .dynamic_obstacles import DynamicObstacleMgr

logger = logging.getLogger(__name__)

Point: TypeAlias = tuple[int, int]


class MultiRobotCoordinatorLike(Protocol):
    """Structural type: avoids hard dependency while keeping type safety."""
    robots: list[RobotAgent]
    robot_goals: dict[int, Point]

    def step_all_robots(self, obstacle_mgr: Optional[DynamicObstacleMgr] = None) -> None: ...
    def get_coordination_statistics(self) -> dict: ...


# Colors (uint8 RGB) — faster than float rendering
FREE = np.array([255, 255, 255], dtype=np.uint8)
STATIC_OBS = np.array([0, 0, 0], dtype=np.uint8)
DYNAMIC_OBS = np.array([255, 0, 0], dtype=np.uint8)
PATH = np.array([179, 230, 255], dtype=np.uint8)   # ~ (0.7, 0.9, 1.0)
GOAL = np.array([0, 255, 0], dtype=np.uint8)
ROBOT = np.array([0, 0, 255], dtype=np.uint8)


@dataclass
class Simulator:
    grid: GridMap
    agent: Optional[RobotAgent] = None
    multi_robot_coordinator: Optional[MultiRobotCoordinatorLike] = None
    obstacle_mgr: Optional[DynamicObstacleMgr] = None

    step_delay: float = 0.1
    max_steps: int = 1000

    def __post_init__(self) -> None:
        if self.step_delay <= 0:
            raise ValueError("step_delay must be > 0")

        self.current_step = 0
        self.running = True

        self.is_multi_robot = self.multi_robot_coordinator is not None
        if not self.is_multi_robot and self.agent is None:
            raise ValueError("Either agent or multi_robot_coordinator must be provided")

        # Visualization setup
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.im = None
        self.anim: Optional[animation.FuncAnimation] = None

        # Print-once guard
        self._goal_reached_printed = False

        # set once; don’t do tight_layout() every frame
        self._layout_configured = False

    # ----------------------------
    # Public API
    # ----------------------------

    def run(self, visualize: bool = True, save_gif: bool = False, gif_filename: str = "simulation.gif") -> None:
        if visualize:
            self.anim = animation.FuncAnimation(
                self.fig,
                self._animate_step,
                frames=self.max_steps,
                interval=self.step_delay * 1000,
                repeat=False,
                blit=False,
            )

            if save_gif:
                fps = max(1, int(round(1.0 / self.step_delay)))
                self.anim.save(gif_filename, writer="pillow", fps=fps)
                print(f"Animation saved as {gif_filename}")
            else:
                plt.show()
        else:
            for step in range(self.max_steps):
                if self._simulation_step():
                    break
                if step % 100 == 0:
                    self._print_progress(step)

    def step_once(self) -> bool:
        """Execute a single simulation step. Returns True if sim should stop."""
        return self._simulation_step()

    def reset(self) -> None:
        """Reset simulation counters (does NOT automatically reset agent/obstacles)."""
        self.current_step = 0
        self.running = True
        self._goal_reached_printed = False

    def get_statistics(self) -> dict:
        if self.is_multi_robot:
            assert self.multi_robot_coordinator is not None
            stats = self.multi_robot_coordinator.get_coordination_statistics()
            return {
                "current_step": self.current_step,
                "mode": "multi",
                **stats,
                "dynamic_obstacles_count": len(self.obstacle_mgr.get_obstacle_positions()) if self.obstacle_mgr else 0,
            }

        assert self.agent is not None
        return {
            "current_step": self.current_step,
            "mode": "single",
            "robot_position": self.agent.position,
            "goal_position": self.agent.goal,
            "goal_reached": self.agent.is_at_goal(),
            "path_length": len(self.agent.path),
            "remaining_path_length": len(self.agent.get_remaining_path()),
            "dynamic_obstacles_count": len(self.obstacle_mgr.get_obstacle_positions()) if self.obstacle_mgr else 0,
        }

    # ----------------------------
    # Animation / loop internals
    # ----------------------------

    def _animate_step(self, _frame):
        if self._simulation_step():
            self.running = False
            if self.anim:
                self.anim.event_source.stop()
            return []
        self._render()
        return []

    def _simulation_step(self) -> bool:
        """Execute one simulation tick. Returns True if simulation should stop."""
        # Update dynamic obstacles first
        if self.obstacle_mgr:
            self.obstacle_mgr.update()

        # Move robots
        if self.is_multi_robot:
            assert self.multi_robot_coordinator is not None
            self.multi_robot_coordinator.step_all_robots(self.obstacle_mgr)

            stats = self.multi_robot_coordinator.get_coordination_statistics()
            if stats.get("total_robots", 0) > 0 and stats.get("robots_at_goal", 0) == stats["total_robots"]:
                if not self._goal_reached_printed:
                    print(f"All robots reached their goals at step {self.current_step}!")
                    self._goal_reached_printed = True
                self.running = False
                return True
        else:
            assert self.agent is not None
            self.agent.step(self.obstacle_mgr)
            if self.agent.is_at_goal():
                if not self._goal_reached_printed:
                    print(f"Goal reached at step {self.current_step}!")
                    self._goal_reached_printed = True
                self.running = False
                return True

        self.current_step += 1
        if self.current_step >= self.max_steps:
            print(f"Maximum steps ({self.max_steps}) reached!")
            self.running = False
            return True

        return False

    def _print_progress(self, step: int) -> None:
        if self.is_multi_robot:
            assert self.multi_robot_coordinator is not None
            stats = self.multi_robot_coordinator.get_coordination_statistics()
            print(f"Step {step}: {stats.get('robots_at_goal', 0)}/{stats.get('total_robots', 0)} robots at goal")
        else:
            assert self.agent is not None
            print(f"Step {step}: Robot at {self.agent.position}")

    # ----------------------------
    # Rendering
    # ----------------------------

    def _render(self) -> None:
        """
        Render current state into an RGB image array then update imshow.
        """
        vis = np.empty((self.grid.height, self.grid.width, 3), dtype=np.uint8)
        vis[:] = FREE

        # static obstacles
        for x, y in self.grid.static_obstacles:
            if 0 <= x < self.grid.width and 0 <= y < self.grid.height:
                vis[y, x] = STATIC_OBS

        # dynamic obstacles
        if self.obstacle_mgr:
            for x, y in self.obstacle_mgr.get_obstacle_positions():
                if 0 <= x < self.grid.width and 0 <= y < self.grid.height:
                    vis[y, x] = DYNAMIC_OBS

        if self.is_multi_robot:
            assert self.multi_robot_coordinator is not None
            coord = self.multi_robot_coordinator

            # paths
            for robot in coord.robots:
                for x, y in robot.get_remaining_path():
                    if 0 <= x < self.grid.width and 0 <= y < self.grid.height:
                        vis[y, x] = PATH

            # goals
            for _, (gx, gy) in coord.robot_goals.items():
                if 0 <= gx < self.grid.width and 0 <= gy < self.grid.height:
                    vis[gy, gx] = GOAL

            # robots on top
            for robot in coord.robots:
                rx, ry = robot.position
                if 0 <= rx < self.grid.width and 0 <= ry < self.grid.height:
                    vis[ry, rx] = ROBOT

        else:
            assert self.agent is not None

            # path
            for x, y in self.agent.get_remaining_path():
                if 0 <= x < self.grid.width and 0 <= y < self.grid.height:
                    vis[y, x] = PATH

            # goal
            if self.agent.goal:
                gx, gy = self.agent.goal
                if 0 <= gx < self.grid.width and 0 <= gy < self.grid.height:
                    vis[gy, gx] = GOAL

            # robot
            rx, ry = self.agent.position
            if 0 <= rx < self.grid.width and 0 <= ry < self.grid.height:
                vis[ry, rx] = ROBOT

        # draw/update
        if self.im is None:
            self.im = self.ax.imshow(vis, origin="upper")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self._setup_grid_lines()
            self._setup_legend()
            self.ax.set_title("AMR Path Planning Simulation")

            if not self._layout_configured:
                plt.tight_layout()
                self._layout_configured = True
        else:
            self.im.set_data(vis)

        self.ax.set_title(f"AMR Path Planning Simulation - Step {self.current_step}")

    def _setup_grid_lines(self) -> None:
        self.ax.set_xticks(np.arange(-0.5, self.grid.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.grid.height, 1), minor=True)
        self.ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    def _setup_legend(self) -> None:
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=ROBOT / 255.0, label="Robot"),
            plt.Rectangle((0, 0), 1, 1, fc=GOAL / 255.0, label="Goal"),
            plt.Rectangle((0, 0), 1, 1, fc=DYNAMIC_OBS / 255.0, label="Dynamic Obstacle"),
            plt.Rectangle((0, 0), 1, 1, fc=STATIC_OBS / 255.0, label="Static Obstacle"),
            plt.Rectangle((0, 0), 1, 1, fc=PATH / 255.0, label="Path"),
        ]
        self.ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))
