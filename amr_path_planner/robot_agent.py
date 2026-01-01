"""
Robot Agent for autonomous mobile robot path planning.

Responsibilities:
- Track robot state (position, goal)
- Follow a planned path
- Replan when blocked by static or dynamic obstacles
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, TypeAlias

from .path_planner import PathPlanner
from .dynamic_obstacles import DynamicObstacleMgr

Point: TypeAlias = tuple[int, int]
BlockedFn: TypeAlias = Callable[[int, int], bool]

logger = logging.getLogger(__name__)


@dataclass
class RobotAgent:
    position: Point
    planner: PathPlanner

    goal: Optional[Point] = None
    path: list[Point] = field(default_factory=list)
    path_index: int = 0

    replanning_needed: bool = False
    replan_cooldown_steps: int = 0
    replan_cooldown_default: int = 2
    max_replans_per_step: int = 1

    # ----------------------------
    # Planning
    # ----------------------------

    def plan_to(self, goal: Point, obstacle_mgr: Optional[DynamicObstacleMgr] = None) -> None:
        self.goal = goal

        blocked_fn = None
        if obstacle_mgr is not None:
            blocked_fn = obstacle_mgr.is_collision

        path = self.planner.compute_path(self.position, goal, blocked_fn=blocked_fn)
        self.path = self._normalize_path(path)
        self.path_index = 0
        self.replanning_needed = False

        if not self.path:
            logger.warning("No path found from %s to %s", self.position, goal)

    # ----------------------------
    # Execution
    # ----------------------------

    def step(self, obstacle_mgr: Optional[DynamicObstacleMgr] = None) -> None:
        if self.goal is None or self.is_at_goal():
            return

        if not self.has_path():
            self._maybe_replan(obstacle_mgr)
            return

        if self.replan_cooldown_steps > 0:
            self.replan_cooldown_steps -= 1
            return

        next_pos = self.path[self.path_index + 1]

        if self._is_blocked(next_pos, obstacle_mgr):
            self.replanning_needed = True
            self._maybe_replan(obstacle_mgr)
            return

        # Move
        self.position = next_pos
        self.path_index += 1
        self.replanning_needed = False

    # ----------------------------
    # Helpers
    # ----------------------------

    def _is_blocked(self, p: Point, obstacle_mgr: Optional[DynamicObstacleMgr]) -> bool:
        x, y = p
        if not self.planner.grid or not self.planner.grid.is_free(x, y):
            return True
        if obstacle_mgr and obstacle_mgr.is_collision(x, y):
            return True
        return False

    def _maybe_replan(self, obstacle_mgr: Optional[DynamicObstacleMgr]) -> None:
        if self.goal is None:
            return

        for _ in range(self.max_replans_per_step):
            self.plan_to(self.goal, obstacle_mgr)
            if self.has_path():
                next_pos = self.path[self.path_index + 1]
                if obstacle_mgr and self._is_blocked(next_pos, obstacle_mgr):
                    self.replan_cooldown_steps = self.replan_cooldown_default
                return

        self.replan_cooldown_steps = self.replan_cooldown_default

    def _normalize_path(self, path: list[Point]) -> list[Point]:
        if not path:
            return []

        dedup = [path[0]]
        for p in path[1:]:
            if p != dedup[-1]:
                dedup.append(p)

        if dedup[0] != self.position:
            dedup.insert(0, self.position)

        return dedup

    # ----------------------------
    # Public helpers
    # ----------------------------

    def is_at_goal(self) -> bool:
        return self.goal is not None and self.position == self.goal

    def has_path(self) -> bool:
        return bool(self.path) and self.path_index < len(self.path) - 1

    def get_remaining_path(self) -> list[Point]:
        return self.path[self.path_index :] if self.path else []

    def set_position(self, position: Point) -> None:
        """Hard reset of robot pose.

        This method is intended for external teleports/resets.
        It intentionally clears any planned path.
        For normal execution along an existing plan, prefer move_to().
        """
        self.position = position
        self.path.clear()
        self.path_index = 0
        self.replanning_needed = False

    def move_to(self, position: Point) -> None:
        """Move robot without clearing its current plan.

        Used by multi-robot coordination logic when a robot advances
        one time step along its already-planned trajectory.
        """
        if position == self.position:
            return

        # If we're following the current plan and the move matches the next step,
        # advance the path index.
        if self.has_path() and self.path_index + 1 < len(self.path) and self.path[self.path_index + 1] == position:
            self.path_index += 1

        self.position = position
        self.replanning_needed = False

    def clear_path(self) -> None:
        self.path.clear()
        self.goal = None
        self.path_index = 0
        self.replanning_needed = False

    def force_replan(self) -> None:
        if self.goal is not None:
            self.plan_to(self.goal)
