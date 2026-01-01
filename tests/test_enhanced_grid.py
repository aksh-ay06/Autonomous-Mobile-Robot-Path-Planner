"""
Test suite for enhanced grid movement models.

UPDATED to match the *actual* EnhancedGridMap implementation in this repo:
- Treats movement model as string: '4-connected', '8-connected', 'knight', 'custom'
- MovementType enum + factory helpers may or may not exist in your build, so tests are conditional.
- Uses EnhancedGridMap public API in this repo:
  - neighbors(x, y)
  - set_cell_cost(x, y, cost) / get_cell_cost(x, y)
  - get_movement_cost(from_pos, to_pos)
  - add_obstacle/remove_obstacle/is_free
- Avoids old API:
  - movement_type / set_movement_type / get_neighbors
  - terrain_costs ndarray / set_terrain_cost / get_terrain_cost
  - create_grid_with_movement / create_terrain_grid (tested only if present)
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any, List, Tuple

import numpy as np

# Add the parent directory to the path to import amr_path_planner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_path_planner.enhanced_grid import EnhancedGridMap

Point = Tuple[int, int]

# Optional features (enum + factories) — only test if they exist
try:
    from amr_path_planner.enhanced_grid import MovementType  # type: ignore
except Exception:
    MovementType = None  # type: ignore

try:
    from amr_path_planner.enhanced_grid import create_grid_with_movement, create_terrain_grid  # type: ignore
except Exception:
    create_grid_with_movement = None  # type: ignore
    create_terrain_grid = None  # type: ignore


def _is_in_bounds(grid: EnhancedGridMap, p: Point) -> bool:
    return 0 <= p[0] < grid.width and 0 <= p[1] < grid.height


class TestMovementTypeEnum(unittest.TestCase):
    """Test MovementType enum (only if present in this build)."""

    def test_movement_type_values_if_present(self) -> None:
        if MovementType is None:
            self.skipTest("MovementType enum not present in this build")

        # Only assert members that actually exist in the enum.
        # (Some builds won't have KING/HEX.)
        expected = {
            "FOUR_CONNECTED": "4-connected",
            "EIGHT_CONNECTED": "8-connected",
            "KNIGHT": "knight",
            "CUSTOM": "custom",
        }

        for member, value in expected.items():
            if hasattr(MovementType, member):
                self.assertEqual(getattr(MovementType, member).value, value)


class TestEnhancedGridMovementModels(unittest.TestCase):
    """Test EnhancedGridMap movement models using the repo's real API."""

    def setUp(self) -> None:
        self.grid = EnhancedGridMap(10, 10)

    def test_initialization(self) -> None:
        self.assertEqual(self.grid.width, 10)
        self.assertEqual(self.grid.height, 10)
        self.assertIsInstance(self.grid.movement_model, str)

        # Default model is commonly '4-connected'
        # (If your default differs, keep this flexible.)
        self.assertIn(self.grid.movement_model, {"4-connected", "8-connected", "knight", "custom"})

        # Default cell cost should be 1.0
        self.assertEqual(self.grid.get_cell_cost(0, 0), 1.0)

    def test_movement_model_setting(self) -> None:
        for model in ("4-connected", "8-connected"):
            with self.subTest(model=model):
                g = EnhancedGridMap(10, 10, movement_model=model)
                self.assertEqual(g.movement_model, model)

        # Optional movement models
        try:
            gk = EnhancedGridMap(10, 10, movement_model="knight")
            self.assertEqual(gk.movement_model, "knight")
        except ValueError:
            pass

        # Invalid
        with self.assertRaises(ValueError):
            EnhancedGridMap(10, 10, movement_model="invalid")

    def test_custom_movement_pattern_if_supported(self) -> None:
        custom_moves = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        try:
            g = EnhancedGridMap(10, 10, movement_model="custom", custom_moves=custom_moves)
        except ValueError:
            self.skipTest("custom movement model not supported in this build")

        self.assertEqual(g.movement_model, "custom")

        # Neighbors from center should match custom deltas
        neighbors = g.neighbors(5, 5)
        expected = {(5, 7), (7, 5), (5, 3), (3, 5)}
        self.assertEqual(set(neighbors), expected)

    def test_four_connected_neighbors(self) -> None:
        g = EnhancedGridMap(10, 10, movement_model="4-connected")

        neighbors = g.neighbors(5, 5)
        expected = {(6, 5), (4, 5), (5, 6), (5, 4)}
        self.assertEqual(set(neighbors), expected)

        # Corner
        corner = g.neighbors(0, 0)
        self.assertEqual(set(corner), {(1, 0), (0, 1)})

        # Bounds check
        for p in neighbors + corner:
            self.assertTrue(_is_in_bounds(g, p))

    def test_eight_connected_neighbors(self) -> None:
        g = EnhancedGridMap(10, 10, movement_model="8-connected")

        neighbors = g.neighbors(5, 5)
        self.assertEqual(len(neighbors), 8)

        expected = {
            (4, 4), (4, 5), (4, 6),
            (5, 4),         (5, 6),
            (6, 4), (6, 5), (6, 6)
        }
        self.assertEqual(set(neighbors), expected)

    def test_knight_neighbors_if_supported(self) -> None:
        try:
            g = EnhancedGridMap(10, 10, movement_model="knight")
        except ValueError:
            self.skipTest("knight movement model not supported in this build")

        neighbors = g.neighbors(5, 5)
        expected = {(3, 4), (3, 6), (4, 3), (4, 7), (6, 3), (6, 7), (7, 4), (7, 6)}
        self.assertEqual(set(neighbors), expected)

        # Edge has fewer
        edge_neighbors = g.neighbors(1, 1)
        self.assertLess(len(edge_neighbors), 8)

    def test_neighbors_filter_obstacles(self) -> None:
        obstacles = {(6, 5), (4, 5)}
        g = EnhancedGridMap(10, 10, static_obstacles=obstacles, movement_model="4-connected")
        neighbors = g.neighbors(5, 5)
        for obs in obstacles:
            self.assertNotIn(obs, neighbors)

    def test_cell_cost_set_get(self) -> None:
        self.assertEqual(self.grid.get_cell_cost(3, 4), 1.0)
        self.grid.set_cell_cost(3, 4, 2.5)
        self.assertEqual(self.grid.get_cell_cost(3, 4), 2.5)

    def test_movement_cost_positive_and_terrain_sensitive(self) -> None:
        g = EnhancedGridMap(10, 10, movement_model="4-connected")
        a: Point = (5, 5)
        b: Point = (5, 6)

        self.assertTrue(g.is_free(*a))
        self.assertTrue(g.is_free(*b))

        base = g.get_movement_cost(a, b)
        self.assertGreater(base, 0.0)

        # Increase destination cost -> movement cost should not decrease
        g.set_cell_cost(b[0], b[1], 3.0)
        higher = g.get_movement_cost(a, b)
        self.assertGreaterEqual(higher, base)

    def test_bounds_checking_all_models(self) -> None:
        models = ["4-connected", "8-connected"]
        # include knight if supported
        try:
            EnhancedGridMap(10, 10, movement_model="knight")
            models.append("knight")
        except ValueError:
            pass

        test_positions = [(0, 0), (0, 9), (9, 0), (9, 9), (0, 5), (9, 5), (5, 0), (5, 9)]
        for model in models:
            g = EnhancedGridMap(10, 10, movement_model=model)
            for x, y in test_positions:
                neighbors = g.neighbors(x, y)
                for nx, ny in neighbors:
                    self.assertGreaterEqual(nx, 0)
                    self.assertLess(nx, g.width)
                    self.assertGreaterEqual(ny, 0)
                    self.assertLess(ny, g.height)

    def test_obstacle_behavior(self) -> None:
        g = EnhancedGridMap(10, 10)
        g.add_obstacle(3, 3)
        g.add_obstacle(4, 4)

        self.assertFalse(g.is_free(3, 3))
        self.assertFalse(g.is_free(4, 4))
        self.assertTrue(g.is_free(5, 5))

        g.remove_obstacle(3, 3)
        self.assertTrue(g.is_free(3, 3))

    def test_neighbor_filtering_with_obstacles(self) -> None:
        g = EnhancedGridMap(10, 10, movement_model="4-connected")
        g.add_obstacle(5, 6)
        neighbors = g.neighbors(5, 5)
        self.assertNotIn((5, 6), neighbors)


class TestFactoryFunctionsIfPresent(unittest.TestCase):
    """Test factory helpers if they exist in your build."""

    def test_create_grid_with_movement_if_present(self) -> None:
        if create_grid_with_movement is None or MovementType is None:
            self.skipTest("Factory create_grid_with_movement or MovementType not present")

        # Prefer known enum members; skip if not available
        if not hasattr(MovementType, "EIGHT_CONNECTED"):
            self.skipTest("MovementType.EIGHT_CONNECTED not present")

        grid = create_grid_with_movement(15, 20, MovementType.EIGHT_CONNECTED)
        self.assertIsInstance(grid, EnhancedGridMap)
        self.assertEqual(grid.width, 15)
        self.assertEqual(grid.height, 20)
        # In this repo, movement_model is a string; factory should map enum->string
        self.assertEqual(grid.movement_model, "8-connected")

    def test_create_terrain_grid_if_present(self) -> None:
        if create_terrain_grid is None or MovementType is None:
            self.skipTest("Factory create_terrain_grid or MovementType not present")

        if not hasattr(MovementType, "FOUR_CONNECTED"):
            self.skipTest("MovementType.FOUR_CONNECTED not present")

        terrain = np.random.uniform(0.5, 3.0, (10, 15))
        grid = create_terrain_grid(terrain, MovementType.FOUR_CONNECTED)
        self.assertIsInstance(grid, EnhancedGridMap)
        self.assertEqual(grid.width, 10)
        self.assertEqual(grid.height, 15)

        # We can't assume internal storage is a numpy array in this repo;
        # but we can validate that get_cell_cost returns expected values.
        self.assertAlmostEqual(grid.get_cell_cost(0, 0), float(terrain[0, 0]), places=6)
        self.assertAlmostEqual(grid.get_cell_cost(9, 14), float(terrain[9, 14]), places=6)

    def test_create_terrain_grid_with_list_if_present(self) -> None:
        if create_terrain_grid is None or MovementType is None:
            self.skipTest("Factory create_terrain_grid or MovementType not present")

        terrain_list = [[1.0, 2.0, 1.5], [2.5, 1.0, 3.0]]  # shape (2,3)
        grid = create_terrain_grid(terrain_list, getattr(MovementType, "FOUR_CONNECTED", "4-connected"))
        self.assertIsInstance(grid, EnhancedGridMap)
        self.assertEqual(grid.width, 2)
        self.assertEqual(grid.height, 3)
        self.assertAlmostEqual(grid.get_cell_cost(0, 0), 1.0, places=6)
        self.assertAlmostEqual(grid.get_cell_cost(1, 2), 3.0, places=6)


class TestMovementIntegration(unittest.TestCase):
    """Integration tests for movement models behavior (no path-planner dependency)."""

    def setUp(self) -> None:
        self.grid = EnhancedGridMap(8, 8)
        for i in range(2, 6):
            self.grid.add_obstacle(i, 4)

    def test_movement_affects_connectivity(self) -> None:
        # 4-connected neighbors
        self.grid.movement_model = "4-connected"
        n4 = self.grid.neighbors(1, 1)

        # 8-connected neighbors
        self.grid.movement_model = "8-connected"
        n8 = self.grid.neighbors(1, 1)

        self.assertGreater(len(n8), len(n4))

    def test_terrain_costs_affect_movement_cost(self) -> None:
        a: Point = (1, 1)
        hi: Point = (2, 2)
        lo: Point = (3, 3)

        self.grid.movement_model = "8-connected"

        # Make sure these are free cells
        self.assertTrue(self.grid.is_free(*hi))
        self.assertTrue(self.grid.is_free(*lo))

        self.grid.set_cell_cost(hi[0], hi[1], 5.0)
        self.grid.set_cell_cost(lo[0], lo[1], 0.1)

        high_cost = self.grid.get_movement_cost(a, hi)
        low_cost = self.grid.get_movement_cost(a, lo)

        self.assertGreater(high_cost, low_cost)

    def test_knight_movement_can_jump_over_obstacles_if_supported(self) -> None:
        try:
            self.grid.movement_model = "knight"
            # Force a neighbors call to confirm support
            _ = self.grid.neighbors(4, 4)
        except Exception:
            self.skipTest("knight movement not supported in this build")

        center = (4, 4)
        surrounding = [
            (3, 3), (3, 4), (3, 5),
            (4, 3),         (4, 5),
            (5, 3), (5, 4), (5, 5),
        ]
        for x, y in surrounding:
            if 0 <= x < 8 and 0 <= y < 8:
                self.grid.add_obstacle(x, y)

        neighbors = self.grid.neighbors(*center)
        self.assertGreater(len(neighbors), 0)

        for nx, ny in neighbors:
            # Knight moves shouldn't be adjacent (Manhattan distance > 2)
            self.assertGreater(abs(nx - center[0]) + abs(ny - center[1]), 2)


if __name__ == "__main__":
    unittest.main()

