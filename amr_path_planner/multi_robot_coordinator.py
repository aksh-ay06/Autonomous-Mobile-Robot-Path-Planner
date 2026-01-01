"""Multi-robot coordination.

This module provides lightweight multi-robot coordination on a shared grid.

Key upgrades (vs. a purely spatial + greedy coordinator):
- Uses *space-time A\** (reservation-table A*) to plan collision-free trajectories.
- Maintains explicit vertex and edge reservations across a configurable horizon.
- Fixes execution bug where per-step movement cleared robot plans.

Coordination modes
------------------
priority:
    Plan robots in priority order (higher priority reserved first).

cooperative:
    Start with priority planning, detect conflicts, and iteratively replan the
    lowest priority robot involved in a conflict.

centralized:
    A lightweight fallback that randomizes priorities when deadlocked.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypeAlias

from .dynamic_obstacles import DynamicObstacleMgr
from .grid_map import GridMap
from .robot_agent import RobotAgent
from .search_algorithms import astar_space_time

Point: TypeAlias = tuple[int, int]
CoordinationMode = str  # "priority" | "cooperative" | "centralized"


@dataclass
class MultiRobotCoordinator:
    grid: GridMap
    coordination_mode: CoordinationMode = "priority"
    horizon: int = 15
    max_coop_iterations: int = 10
    centralized_retries: int = 5

    robots: list[RobotAgent] = field(default_factory=list)
    robot_goals: dict[int, Point] = field(default_factory=dict)
    robot_priorities: dict[int, int] = field(default_factory=dict)

    # Time-indexed plans for each robot, aligned to the *current* time step.
    # plans[rid][0] is always the robot's current position.
    plans: dict[int, list[Point]] = field(default_factory=dict)

    # Global reservation tables in *absolute* time.
    # vertex: (t_abs, cell) -> rid
    reserved_v: dict[tuple[int, Point], int] = field(default_factory=dict)
    # edge: (t_abs, u, v) -> rid  (move u->v between t_abs and t_abs+1)
    reserved_e: dict[tuple[int, Point, Point], int] = field(default_factory=dict)

    # Absolute simulation time maintained by coordinator.
    time_step: int = 0

    # ----------------------------
    # Public API
    # ----------------------------

    def add_robot(self, robot: RobotAgent, priority: int = 0) -> int:
        rid = len(self.robots)
        self.robots.append(robot)
        self.robot_priorities[rid] = priority
        self.plans[rid] = [robot.position]
        return rid

    def set_robot_goal(self, robot_id: int, goal: Point) -> None:
        self._check_robot_id(robot_id)
        self.robot_goals[robot_id] = goal

    def plan_coordinated_paths(self, obstacle_mgr: Optional[DynamicObstacleMgr] = None) -> None:
        """(Re)plan collision-free trajectories for all robots."""
        if self.coordination_mode == "priority":
            self._plan_priority(obstacle_mgr)
        elif self.coordination_mode == "cooperative":
            self._plan_cooperative(obstacle_mgr)
        else:
            self._plan_centralized(obstacle_mgr)

    def step_all_robots(self, obstacle_mgr: Optional[DynamicObstacleMgr] = None) -> None:
        """Advance all robots by one time step along their reserved trajectories.

        This function assumes you planned first via plan_coordinated_paths().
        If a robot has no viable plan (e.g., dynamic obstacle blocks), it will wait.
        """
        # If any robot is missing a plan (or dynamic obstacles changed), do a quick replan.
        if any((rid in self.robot_goals and len(self.plans.get(rid, [])) <= 1 and not self.robots[rid].is_at_goal()) for rid in range(len(self.robots))):
            self.plan_coordinated_paths(obstacle_mgr)

        proposals: dict[int, Point] = {}
        for rid, robot in enumerate(self.robots):
            proposals[rid] = self._proposed_next(rid, obstacle_mgr)

        final_moves = self._resolve_step_conflicts(proposals)

        # Execute moves without wiping plans.
        for rid, robot in enumerate(self.robots):
            nxt = final_moves[rid]
            robot.move_to(nxt)

        # Advance coordinator time and shift plan windows.
        self.time_step += 1
        for rid in range(len(self.robots)):
            if self.plans.get(rid):
                # Drop the first (previous current position) to keep alignment.
                if self.plans[rid][0] != self.robots[rid].position:
                    # If execution diverged, hard reset the window.
                    self.plans[rid] = [self.robots[rid].position]
                elif len(self.plans[rid]) > 1:
                    self.plans[rid] = self.plans[rid][1:]
                else:
                    self.plans[rid] = [self.robots[rid].position]

        # Optional: if dynamic obstacles exist, you may want to replan periodically.

    def get_coordination_statistics(self) -> dict:
        total = len(self.robots)
        at_goal = sum(1 for r in self.robots if r.is_at_goal())
        return {
            "total_robots": total,
            "robots_at_goal": at_goal,
            "robots_with_plans": sum(1 for rid in range(total) if len(self.plans.get(rid, [])) > 1),
            "coordination_mode": self.coordination_mode,
            "time_step": self.time_step,
            "completion_rate": at_goal / total if total else 0.0,
        }

    # ----------------------------
    # Planning modes
    # ----------------------------

    def _plan_priority(self, obstacle_mgr: Optional[DynamicObstacleMgr]) -> None:
        # Clear all future reservations (>= current time) and rebuild from scratch.
        self._clear_all_future_reservations()

        for rid, _ in self._robots_by_priority():
            self._replan_one(rid, obstacle_mgr)

    def _plan_cooperative(self, obstacle_mgr: Optional[DynamicObstacleMgr]) -> None:
        self._plan_priority(obstacle_mgr)
        for _ in range(self.max_coop_iterations):
            conflicts = self._detect_conflicts()
            if not conflicts:
                return
            low_priority = min(conflicts, key=lambda r: self.robot_priorities.get(r, 0))
            self._replan_one(low_priority, obstacle_mgr)

    def _plan_centralized(self, obstacle_mgr: Optional[DynamicObstacleMgr]) -> None:
        self._plan_priority(obstacle_mgr)
        for _ in range(self.centralized_retries):
            if not self._has_deadlock():
                return
            ids = list(range(len(self.robots)))
            random.shuffle(ids)
            for i, rid in enumerate(ids):
                self.robot_priorities[rid] = len(ids) - i
            self._plan_priority(obstacle_mgr)

    # ----------------------------
    # Core planning
    # ----------------------------

    def _replan_one(self, robot_id: int, obstacle_mgr: Optional[DynamicObstacleMgr]) -> None:
        """Plan a single robot with reservations from higher-priority robots."""
        if robot_id not in self.robot_goals:
            self.plans[robot_id] = [self.robots[robot_id].position]
            return

        robot = self.robots[robot_id]
        goal = self.robot_goals[robot_id]

        # Clear this robot's existing future reservations.
        self._clear_robot_future_reservations(robot_id)

        # Build local reservation tables relative to current time (t=0..horizon).
        local_v: dict[tuple[int, Point], int] = {}
        local_e: dict[tuple[int, Point, Point], int] = {}
        for (t_abs, cell), owner in self.reserved_v.items():
            if t_abs < self.time_step or t_abs > self.time_step + self.horizon:
                continue
            local_v[(t_abs - self.time_step, cell)] = owner

        for (t_abs, u, v), owner in self.reserved_e.items():
            if t_abs < self.time_step or t_abs > self.time_step + self.horizon:
                continue
            local_e[(t_abs - self.time_step, u, v)] = owner

        # Treat dynamic obstacles as always-blocked in planning horizon (static approximation).
        # If you have predictable obstacle trajectories, this is the place to index them by time.
        blocked_cells: set[Point] = set()
        if obstacle_mgr:
            blocked_cells.update(obstacle_mgr.get_obstacle_positions())

        def is_free_with_dynamic(x: int, y: int) -> bool:
            return self.grid.is_free(x, y) and (x, y) not in blocked_cells

        # Wrapper that preserves the base grid's movement / cost APIs while filtering dynamics.
        class _GridView:
            def __init__(self, base: GridMap):
                self._base = base
                self.width = base.width
                self.height = base.height
                self.static_obstacles = getattr(base, "static_obstacles", set())

            def is_free(self, x: int, y: int) -> bool:
                return self._base.is_free(x, y) and (x, y) not in blocked_cells

            def neighbors4(self, x: int, y: int):
                if hasattr(self._base, "neighbors4"):
                    nbs = self._base.neighbors4(x, y)  # type: ignore[attr-defined]
                else:
                    candidates = ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y))
                    nbs = candidates
                return [p for p in nbs if self.is_free(p[0], p[1])]

            def neighbors(self, x: int, y: int):
                if hasattr(self._base, "neighbors"):
                    nbs = self._base.neighbors(x, y)  # type: ignore[attr-defined]
                else:
                    nbs = self.neighbors4(x, y)
                return [p for p in nbs if self.is_free(p[0], p[1])]

            def neighbors_with_cost(self, x: int, y: int):
                if hasattr(self._base, "neighbors_with_cost"):
                    pairs = self._base.neighbors_with_cost(x, y)  # type: ignore[attr-defined]
                    return [(p, c) for (p, c) in pairs if self.is_free(p[0], p[1])]
                return [(p, 1.0) for p in self.neighbors(x, y)]

        grid_view = _GridView(self.grid)

        path = astar_space_time(
            start=robot.position,
            goal=goal,
            grid=grid_view,
            reserved_v=local_v,
            reserved_e=local_e,
            robot_id=robot_id,
            horizon=self.horizon,
            allow_wait=True,
        )

        if not path:
            # No feasible path within horizon: reserve a wait at current cell.
            self.plans[robot_id] = [robot.position]
            self._reserve_vertex(robot_id, self.time_step, robot.position)
            return

        # Ensure plan window begins at current position.
        if path[0] != robot.position:
            path = [robot.position] + path

        # Pad to horizon+1 with goal waits, so reservations remain stable.
        if len(path) < self.horizon + 1:
            tail = path[-1]
            path = path + [tail] * (self.horizon + 1 - len(path))
        else:
            path = path[: self.horizon + 1]

        self.plans[robot_id] = path

        # Push into robot's internal plan so RobotAgent.has_path() works.
        robot.path = path
        robot.path_index = 0
        robot.goal = goal

        # Reserve vertices and edges in absolute time.
        for t in range(len(path)):
            self._reserve_vertex(robot_id, self.time_step + t, path[t])
            if t + 1 < len(path) and path[t + 1] != path[t]:
                self._reserve_edge(robot_id, self.time_step + t, path[t], path[t + 1])

    # ----------------------------
    # Conflict handling
    # ----------------------------

    def _proposed_next(self, robot_id: int, obstacle_mgr: Optional[DynamicObstacleMgr]) -> Point:
        robot = self.robots[robot_id]

        # Prefer coordinator plan window.
        plan = self.plans.get(robot_id, [])
        if len(plan) >= 2:
            nxt = plan[1]
        elif robot.has_path():
            nxt = robot.get_remaining_path()[1]
        else:
            nxt = robot.position

        # If dynamic obstacle blocks the move, wait and trigger replanning.
        if obstacle_mgr and obstacle_mgr.is_collision(*nxt):
            return robot.position
        return nxt

    def _resolve_step_conflicts(self, proposals: dict[int, Point]) -> dict[int, Point]:
        """Resolve vertex + edge-swap conflicts for a single step.

        Policy: higher priority wins; lower priority waits.
        """
        current = {rid: self.robots[rid].position for rid in range(len(self.robots))}

        # Track accepted destinations and accepted directed edges for swap avoidance.
        final: dict[int, Point] = dict(current)
        occupied_next: dict[Point, int] = {}
        accepted_edges: set[tuple[Point, Point]] = set()

        # Process in priority order.
        for rid, _ in self._robots_by_priority():
            src = current[rid]
            dst = proposals[rid]

            # Vertex conflict: destination already taken.
            if dst in occupied_next:
                continue

            # Edge swap conflict: higher-priority robot already accepted dst->src.
            if (dst, src) in accepted_edges:
                continue

            final[rid] = dst
            occupied_next[dst] = rid
            accepted_edges.add((src, dst))

        return final

    def _detect_conflicts(self) -> List[int]:
        """Detect robots involved in vertex/edge conflicts across the planning horizon."""
        conflicts: set[int] = set()
        H = self.horizon
        for i in range(len(self.robots)):
            for j in range(i + 1, len(self.robots)):
                pi = self.plans.get(i, [])
                pj = self.plans.get(j, [])
                if not pi or not pj:
                    continue
                T = min(H, len(pi) - 1, len(pj) - 1)
                for t in range(T + 1):
                    if t < len(pi) and t < len(pj) and pi[t] == pj[t]:
                        conflicts.update([i, j])
                    if t + 1 < len(pi) and t + 1 < len(pj):
                        # edge swap
                        if pi[t] == pj[t + 1] and pi[t + 1] == pj[t]:
                            conflicts.update([i, j])
        return sorted(conflicts)

    # ----------------------------
    # Reservation helpers
    # ----------------------------

    def _reserve_vertex(self, robot_id: int, t_abs: int, cell: Point) -> None:
        self.reserved_v[(t_abs, cell)] = robot_id

    def _reserve_edge(self, robot_id: int, t_abs: int, u: Point, v: Point) -> None:
        self.reserved_e[(t_abs, u, v)] = robot_id

    def _clear_robot_future_reservations(self, robot_id: int) -> None:
        tv = [k for k, owner in self.reserved_v.items() if owner == robot_id and k[0] >= self.time_step]
        for k in tv:
            del self.reserved_v[k]

        te = [k for k, owner in self.reserved_e.items() if owner == robot_id and k[0] >= self.time_step]
        for k in te:
            del self.reserved_e[k]

    def _clear_all_future_reservations(self) -> None:
        tv = [k for k in self.reserved_v.keys() if k[0] >= self.time_step]
        for k in tv:
            del self.reserved_v[k]
        te = [k for k in self.reserved_e.keys() if k[0] >= self.time_step]
        for k in te:
            del self.reserved_e[k]

    # ----------------------------
    # Misc
    # ----------------------------

    def _has_deadlock(self) -> bool:
        # Deadlock heuristic: robot has a goal but can't move in the next step window.
        return any(
            rid in self.robot_goals
            and not self.robots[rid].is_at_goal()
            and (len(self.plans.get(rid, [])) <= 1 or self.plans[rid][0] == self.plans[rid][1])
            for rid in range(len(self.robots))
        )

    def _robots_by_priority(self):
        return sorted(
            enumerate(self.robots),
            key=lambda x: self.robot_priorities.get(x[0], 0),
            reverse=True,
        )

    def _check_robot_id(self, robot_id: int) -> None:
        if robot_id < 0 or robot_id >= len(self.robots):
            raise ValueError(f"Robot ID {robot_id} does not exist")
