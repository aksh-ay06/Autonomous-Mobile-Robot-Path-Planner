from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[int, int]


@dataclass
class GridMap:
    """2D grid map with static obstacles. Coordinates are (x, y)."""
    width: int
    height: int
    static_obstacles: set[Point] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive integers.")
        # Avoid aliasing caller-owned set
        self.static_obstacles = set(self.static_obstacles)

        # Strict policy (recommended): fail fast if obstacles are invalid.
        bad = [p for p in self.static_obstacles if not self.in_bounds(*p)]
        if bad:
            raise ValueError(f"Out-of-bounds obstacles: {bad}")

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and (x, y) not in self.static_obstacles

    def neighbors4(self, x: int, y: int) -> list[Point]:
        candidates = ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y))
        return [(nx, ny) for nx, ny in candidates if self.is_free(nx, ny)]

    def neighbors(self, x: int, y: int) -> list[Point]:
        """Alias to 4-connected neighbors (compat with tests)."""
        return self.neighbors4(x, y)

    def add_obstacle(self, x: int, y: int) -> None:
        if not self.in_bounds(x, y):
            raise ValueError(f"Obstacle ({x},{y}) out of bounds.")
        self.static_obstacles.add((x, y))

    def remove_obstacle(self, x: int, y: int) -> None:
        self.static_obstacles.discard((x, y))
