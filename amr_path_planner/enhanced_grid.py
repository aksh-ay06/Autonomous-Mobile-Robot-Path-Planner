"""
Enhanced grid map with support for different movement models and cell costs.

Compatible with:
- GridMap (dataclass) interface: in_bounds(), is_free(), add/remove obstacles
- search_algorithms.py which currently uses neighbors4()

Adds:
- neighbors8()
- neighbors_with_cost() for 4/8/custom move sets (supports terrain costs and diagonal costs)
- movement patterns factory utilities
"""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, TypeAlias

from .grid_map import GridMap

Point: TypeAlias = tuple[int, int]


class MovementType(str, enum.Enum):
    FOUR_CONNECTED = "4-connected"
    EIGHT_CONNECTED = "8-connected"
    CUSTOM = "custom"
    KNIGHT = "knight"  # convenience alias => custom pattern


# Common movement deltas
MOVES_4: tuple[Point, ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))
MOVES_8: tuple[Point, ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),          (0, 1),
    (1, -1),  (1, 0), (1, 1),
)


@dataclass
class EnhancedGridMap(GridMap):
    """
    GridMap with configurable movement model and terrain / cell costs.

    - movement: controls which deltas are allowed
    - custom_moves: deltas used when movement == CUSTOM (or pattern factories)
    - cost_map: per-cell base cost multiplier (>= 0). Default 1.0.
      Movement cost = base_cost(to_cell) * geometric_distance(from,to)
    """
    movement: MovementType = MovementType.FOUR_CONNECTED
    movement_model: str | MovementType | None = None
    custom_moves: tuple[Point, ...] = field(default_factory=tuple)
    cost_map: Dict[Point, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

        # Normalize movement from either movement_model or movement
        model_value = self.movement_model if self.movement_model is not None else self.movement

        # Convert to canonical string then enum for internal use
        if isinstance(model_value, MovementType):
            model_str = model_value.value
        else:
            model_str = str(model_value)

        try:
            m_enum = MovementType(model_str)
        except ValueError as exc:  # invalid model string
            raise ValueError(f"Unknown movement model: {model_str}") from exc

        # Handle knight as a convenience alias
        if m_enum == MovementType.KNIGHT:
            self.movement = MovementType.CUSTOM
            self.movement_model = "knight"
            if not self.custom_moves:
                self.custom_moves = tuple(create_custom_movement_pattern("knight"))
        elif m_enum == MovementType.CUSTOM:
            self.movement = MovementType.CUSTOM
            self.movement_model = "custom"
        else:
            self.movement = m_enum
            self.movement_model = m_enum.value

        if self.movement == MovementType.CUSTOM and not self.custom_moves:
            raise ValueError("custom_moves must be provided when movement model is CUSTOM.")

        # Ensure costs are sane
        for cell, c in list(self.cost_map.items()):
            if c < 0:
                raise ValueError(f"Cell cost must be non-negative. Got {c} for {cell}.")

    # ----------------------------
    # Neighbor APIs (compat + extended)
    # ----------------------------

    def neighbors4(self, x: int, y: int) -> list[Point]:
        """4-connected neighbors (compatible with existing A*/Dijkstra)."""
        return [(x + dx, y + dy) for dx, dy in MOVES_4 if self.is_free(x + dx, y + dy)]

    def neighbors8(self, x: int, y: int) -> list[Point]:
        """8-connected neighbors (includes diagonals)."""
        return [(x + dx, y + dy) for dx, dy in MOVES_8 if self.is_free(x + dx, y + dy)]

    def neighbors(self, x: int, y: int) -> list[Point]:
        """
        Movement-model neighbors:
        - FOUR_CONNECTED -> 4 neighbors
        - EIGHT_CONNECTED -> 8 neighbors
        - CUSTOM -> custom_moves
        """
        moves = self._moves()
        out: list[Point] = []
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if self.is_free(nx, ny):
                out.append((nx, ny))
        return out

    def neighbors_with_cost(self, x: int, y: int) -> list[tuple[Point, float]]:
        """
        Neighbors + movement costs based on:
        cost = base_cost(to_cell) * euclidean_distance(move)
        """
        moves = self._moves()
        out: list[tuple[Point, float]] = []
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if not self.is_free(nx, ny):
                continue
            cost = self.get_movement_cost((x, y), (nx, ny))
            out.append(((nx, ny), cost))
        return out

    def _moves(self) -> tuple[Point, ...]:
        model = self.movement_model if isinstance(self.movement_model, str) else self.movement_model.value

        if model == MovementType.FOUR_CONNECTED.value:
            return MOVES_4
        if model == MovementType.EIGHT_CONNECTED.value:
            return MOVES_8
        if model == MovementType.KNIGHT.value:
            return tuple(create_custom_movement_pattern("knight"))
        if model == MovementType.CUSTOM.value:
            if not self.custom_moves:
                raise ValueError("custom_moves must be provided when movement model is CUSTOM.")
            return tuple(self.custom_moves)
        raise ValueError(f"Unknown movement model: {model}")

    # ----------------------------
    # Costs
    # ----------------------------

    def get_cell_cost(self, x: int, y: int) -> float:
        """Base terrain cost multiplier for entering cell (x,y)."""
        return float(self.cost_map.get((x, y), 1.0))

    def set_cell_cost(self, x: int, y: int, cost: float) -> None:
        if cost < 0:
            raise ValueError("Cell cost must be non-negative.")
        if not self.in_bounds(x, y):
            raise ValueError(f"Cell ({x},{y}) out of bounds.")
        self.cost_map[(x, y)] = float(cost)

    def set_terrain_costs(self, terrain_map: Dict[Point, float]) -> None:
        for (x, y), c in terrain_map.items():
            self.set_cell_cost(x, y, c)

    def get_movement_cost(self, from_pos: Point, to_pos: Point) -> float:
        """
        Movement cost:
          base_cost(to_pos) * euclidean_distance(delta)
        """
        base = self.get_cell_cost(*to_pos)
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        dist = math.hypot(dx, dy)
        return base * dist

    # ----------------------------
    # Terrain patterns
    # ----------------------------

    def create_terrain_pattern(self, pattern_type: str, **kwargs) -> None:
        """
        Create terrain patterns:
          - mud: random high-cost cells
          - hills: radial gradient
          - water: rectangular regions high-cost
          - roads: polyline roads low-cost (Bresenham)
        """
        pattern_type = pattern_type.lower().strip()

        if pattern_type == "mud":
            mud_cost = float(kwargs.get("cost", 2.0))
            density = float(kwargs.get("density", 0.1))
            seed = kwargs.get("seed", None)
            rng = random.Random(seed)
            for x in range(self.width):
                for y in range(self.height):
                    if self.is_free(x, y) and rng.random() < density:
                        self.set_cell_cost(x, y, mud_cost)

        elif pattern_type == "hills":
            cx = int(kwargs.get("center_x", self.width // 2))
            cy = int(kwargs.get("center_y", self.height // 2))
            max_cost = float(kwargs.get("max_cost", 3.0))
            radius = float(kwargs.get("radius", min(self.width, self.height) // 4))
            radius = max(radius, 1.0)

            for x in range(self.width):
                for y in range(self.height):
                    if not self.is_free(x, y):
                        continue
                    d = math.hypot(x - cx, y - cy)
                    if d <= radius:
                        # higher cost near center
                        c = 1.0 + (max_cost - 1.0) * (1.0 - d / radius)
                        self.set_cell_cost(x, y, c)

        elif pattern_type == "water":
            water_cost = float(kwargs.get("cost", 10.0))
            regions = kwargs.get("regions", [])
            # regions: list[(x_start, y_start, w, h)]
            for x0, y0, w, h in regions:
                for x in range(x0, min(x0 + w, self.width)):
                    for y in range(y0, min(y0 + h, self.height)):
                        if self.is_free(x, y):
                            self.set_cell_cost(x, y, water_cost)

        elif pattern_type == "roads":
            road_cost = float(kwargs.get("cost", 0.5))
            roads = kwargs.get("roads", [])
            # roads: list[((x1,y1),(x2,y2)), ...]
            for (x1, y1), (x2, y2) in roads:
                for x, y in _bresenham_cells((x1, y1), (x2, y2)):
                    if self.is_free(x, y):
                        self.set_cell_cost(x, y, road_cost)

        else:
            raise ValueError(f"Unknown terrain pattern: {pattern_type}")


def _bresenham_cells(a: Point, b: Point) -> list[Point]:
    """Bresenham line cells between integer points a and b."""
    x0, y0 = a
    x1, y1 = b
    cells: list[Point] = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    x, y = x0, y0
    if dx >= dy:
        err = dx // 2
        while x != x1:
            cells.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
        cells.append((x1, y1))
    else:
        err = dy // 2
        while y != y1:
            cells.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
        cells.append((x1, y1))

    return cells


# ----------------------------
# Movement pattern utilities
# ----------------------------

def create_custom_movement_pattern(pattern_name: str) -> list[Point]:
    name = pattern_name.lower().strip()

    if name == "king":
        return list(MOVES_8)

    if name == "knight":
        return [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

    if name == "plus":
        return list(MOVES_4)

    if name == "cross":
        return [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    if name == "extended":
        moves: list[Point] = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                moves.append((dx, dy))
        return moves

    if name == "hex":
        return [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    raise ValueError(f"Unknown movement pattern: {pattern_name}")


def create_enhanced_grid(
    width: int,
    height: int,
    movement_type: str = "4-connected",
    obstacles: Optional[set[Point]] = None,
    terrain_config: Optional[Dict] = None,
) -> EnhancedGridMap:
    """
    Factory:
      - movement_type in {"4-connected","8-connected","custom", "king","knight","plus","cross","extended","hex"}
      - terrain_config: { "mud": {...}, "roads": {...}, ... }
    """
    obstacles = obstacles or set()
    m = movement_type.lower().strip()

    if m in {"king", "knight", "plus", "cross", "extended", "hex"}:
        moves = tuple(create_custom_movement_pattern(m))
        grid = EnhancedGridMap(width=width, height=height, static_obstacles=obstacles,
                               movement=MovementType.CUSTOM, custom_moves=moves)
    else:
        # allow "4-connected" / "8-connected" / "custom"
        movement = MovementType(m)
        grid = EnhancedGridMap(width=width, height=height, static_obstacles=obstacles,
                               movement=movement)

    if terrain_config:
        for terrain_type, cfg in terrain_config.items():
            grid.create_terrain_pattern(terrain_type, **cfg)

    return grid
