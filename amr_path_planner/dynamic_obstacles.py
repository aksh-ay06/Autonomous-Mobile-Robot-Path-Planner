"""
Dynamic obstacles manager for autonomous mobile robot path planning.

Maintains a set of dynamic obstacle positions and updates them using
a random-walk style motion model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, TypeAlias

from .grid_map import GridMap

Point: TypeAlias = tuple[int, int]
BlockedFn: TypeAlias = Callable[[int, int], bool]


@dataclass
class DynamicObstacleMgr:
    """
    Manager for dynamic obstacles that move randomly on the grid.

    Notes:
    - Dynamic obstacles are kept as a set for O(1) collision checks.
    - Update is synchronous: all obstacles decide moves, then positions update together.
    """

    grid: GridMap
    movement_probability: float = 0.7
    rng: random.Random = field(default_factory=random.Random)

    # Positions of dynamic obstacles
    obstacles: set[Point] = field(default_factory=set)

    # Optional user-provided blocking rule (in addition to grid/static + other dynamic)
    blocked_fn: Optional[BlockedFn] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.movement_probability <= 1.0):
            raise ValueError("movement_probability must be between 0.0 and 1.0")

        # Defensive copy in case caller passes a set they hold onto
        self.obstacles = set(self.obstacles)

        # Drop out-of-bounds or statically blocked obstacles (safer default)
        self.obstacles = {
            p
            for p in self.obstacles
            if self.grid.in_bounds(*p) and self.grid.is_free(*p)
        }

    # ----------------------------
    # Basic API
    # ----------------------------

    def add_obstacle(self, x: int, y: int) -> bool:
        """
        Add a dynamic obstacle. Returns True if added.
        """
        p = (x, y)
        if not self._cell_free_for_dynamic(x, y):
            return False
        self.obstacles.add(p)
        return True

    def remove_obstacle(self, x: int, y: int) -> bool:
        """
        Remove a dynamic obstacle. Returns True if removed.
        """
        p = (x, y)
        if p in self.obstacles:
            self.obstacles.remove(p)
            return True
        return False

    def clear_obstacles(self) -> None:
        self.obstacles.clear()

    def is_collision(self, x: int, y: int) -> bool:
        return (x, y) in self.obstacles

    def get_obstacle_positions(self) -> list[Point]:
        # stable snapshot (handy for deterministic visualization ordering)
        return sorted(self.obstacles)

    # ----------------------------
    # Spawning
    # ----------------------------

    def spawn_random_obstacles(self, count: int, max_attempts: Optional[int] = None) -> int:
        """
        Spawn up to `count` random obstacles in free cells.
        Returns number of obstacles actually spawned.
        """
        if count <= 0:
            return 0
        if max_attempts is None:
            max_attempts = max(10, count * 10)

        spawned = 0
        attempts = 0
        while spawned < count and attempts < max_attempts:
            x = self.rng.randint(0, self.grid.width - 1)
            y = self.rng.randint(0, self.grid.height - 1)
            if self.add_obstacle(x, y):
                spawned += 1
            attempts += 1

        return spawned

    def seed(self, seed: int) -> None:
        """Seed RNG for reproducible obstacle motion."""
        self.rng.seed(seed)

    # ----------------------------
    # Dynamics
    # ----------------------------

    def update(self) -> None:
        """
        Synchronous update:
        - each obstacle may move (probability movement_probability)
        - picks a random valid destination among candidates
        - ensures no two dynamic obstacles land on the same cell
        """
        if not self.obstacles:
            return

        # We'll build next positions into a new set.
        next_positions: set[Point] = set()

        # Process in random order to avoid bias (still synchronous via next_positions)
        obstacles_list = list(self.obstacles)
        self.rng.shuffle(obstacles_list)

        for (ox, oy) in obstacles_list:
            # default: stay
            chosen = (ox, oy)

            if self.rng.random() < self.movement_probability:
                candidates = self._candidate_moves(ox, oy)

                # filter:
                # - must be free wrt grid/static/custom blocked_fn
                # - must not collide with already-chosen next positions
                # - must not collide with other dynamic obstacles that have not moved yet?
                #   (synchronous model allows them to move away, so we only prevent landing conflicts)
                valid: list[Point] = []
                for nx, ny in candidates:
                    if (nx, ny) in next_positions:
                        continue
                    if not self._cell_free_for_dynamic(nx, ny):
                        continue
                    valid.append((nx, ny))

                if valid:
                    chosen = self.rng.choice(valid)

            # If our chosen is already taken (rare if staying + earlier claimed), force stay if possible
            if chosen in next_positions:
                if (ox, oy) not in next_positions:
                    chosen = (ox, oy)
                else:
                    # ultimate fallback: find any free cell among candidates including stay
                    fallback = [(ox, oy)] + self._candidate_moves(ox, oy)
                    for nx, ny in fallback:
                        if (nx, ny) not in next_positions and self._cell_free_for_dynamic(nx, ny):
                            chosen = (nx, ny)
                            break

            next_positions.add(chosen)

        self.obstacles = next_positions

    # ----------------------------
    # Internals
    # ----------------------------

    def _candidate_moves(self, x: int, y: int) -> list[Point]:
        """
        Candidate moves (4-connected by default).
        If grid exposes neighbors4, reuse it to keep semantics consistent.
        Includes staying in place.
        """
        # if EnhancedGridMap were used, it might not have neighbors4; GridMap does in your refactor.
        if hasattr(self.grid, "neighbors4"):
            # neighbors4 already filters static obstacles and bounds via is_free()
            nbrs = self.grid.neighbors4(x, y)  # type: ignore[attr-defined]
            return [(x, y)] + list(nbrs)

        # fallback 4-connected
        candidates = [(x, y), (x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
        return candidates

    def _cell_free_for_dynamic(self, x: int, y: int) -> bool:
        """
        Checks whether a cell is eligible for a dynamic obstacle to occupy.
        - must be in bounds
        - must not be a static obstacle
        - must satisfy optional blocked_fn
        - note: does NOT check against other dynamic obstacles (caller decides depending on synchronous model)
        """
        if not self.grid.in_bounds(x, y):
            return False
        if not self.grid.is_free(x, y):
            return False
        if self.blocked_fn and self.blocked_fn(x, y):
            return False
        return True
