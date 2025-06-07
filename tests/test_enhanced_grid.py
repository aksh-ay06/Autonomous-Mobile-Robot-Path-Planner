"""
Test suite for enhanced grid movement models.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path to import amr_path_planner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_path_planner.enhanced_grid import (
    EnhancedGridMap, MovementType, 
    create_grid_with_movement, create_terrain_grid
)


class TestMovementType(unittest.TestCase):
    """Test the MovementType enum."""
    
    def test_movement_type_values(self):
        """Test that all movement types have correct values."""
        self.assertEqual(MovementType.FOUR_CONNECTED.value, "4-connected")
        self.assertEqual(MovementType.EIGHT_CONNECTED.value, "8-connected")
        self.assertEqual(MovementType.KING.value, "king")
        self.assertEqual(MovementType.KNIGHT.value, "knight")
        self.assertEqual(MovementType.HEX.value, "hex")
        self.assertEqual(MovementType.CUSTOM.value, "custom")


class TestEnhancedGridMap(unittest.TestCase):
    """Test the EnhancedGridMap class."""
    
    def setUp(self):
        """Set up test environment."""
        self.grid = EnhancedGridMap(10, 10)
    
    def test_initialization(self):
        """Test basic grid initialization."""
        self.assertEqual(self.grid.width, 10)
        self.assertEqual(self.grid.height, 10)
        self.assertEqual(self.grid.movement_type, MovementType.FOUR_CONNECTED)
        self.assertIsInstance(self.grid.terrain_costs, np.ndarray)
        self.assertEqual(self.grid.terrain_costs.shape, (10, 10))
        
        # All terrain costs should be 1.0 initially
        self.assertTrue(np.all(self.grid.terrain_costs == 1.0))
    
    def test_movement_type_setting(self):
        """Test setting different movement types."""
        movement_types = [
            MovementType.FOUR_CONNECTED,
            MovementType.EIGHT_CONNECTED,
            MovementType.KING,
            MovementType.KNIGHT,
            MovementType.HEX
        ]
        
        for movement_type in movement_types:
            with self.subTest(movement_type=movement_type):
                self.grid.set_movement_type(movement_type)
                self.assertEqual(self.grid.movement_type, movement_type)
    
    def test_custom_movement_pattern(self):
        """Test setting custom movement patterns."""
        custom_moves = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # Only cardinal directions, 2 steps
        
        self.grid.set_custom_movement_pattern(custom_moves)
        self.assertEqual(self.grid.movement_type, MovementType.CUSTOM)
        self.assertEqual(self.grid.custom_moves, custom_moves)
    
    def test_four_connected_neighbors(self):
        """Test 4-connected movement neighbors."""
        self.grid.set_movement_type(MovementType.FOUR_CONNECTED)
        
        # Test center position
        neighbors = self.grid.get_neighbors(5, 5)
        expected = [(5, 4), (5, 6), (4, 5), (6, 5)]  # Up, Down, Left, Right
        self.assertEqual(set(neighbors), set(expected))
        
        # Test corner position
        corner_neighbors = self.grid.get_neighbors(0, 0)
        expected_corner = [(0, 1), (1, 0)]  # Only Down and Right
        self.assertEqual(set(corner_neighbors), set(expected_corner))
    
    def test_eight_connected_neighbors(self):
        """Test 8-connected movement neighbors."""
        self.grid.set_movement_type(MovementType.EIGHT_CONNECTED)
        
        # Test center position
        neighbors = self.grid.get_neighbors(5, 5)
        expected = [
            (4, 4), (4, 5), (4, 6),
            (5, 4),         (5, 6),
            (6, 4), (6, 5), (6, 6)
        ]
        self.assertEqual(set(neighbors), set(expected))
        
        # Should have 8 neighbors for center position
        self.assertEqual(len(neighbors), 8)
    
    def test_king_movement_neighbors(self):
        """Test king movement (same as 8-connected)."""
        self.grid.set_movement_type(MovementType.KING)
        
        neighbors = self.grid.get_neighbors(5, 5)
        # King movement is same as 8-connected
        self.assertEqual(len(neighbors), 8)
    
    def test_knight_movement_neighbors(self):
        """Test knight movement neighbors."""
        self.grid.set_movement_type(MovementType.KNIGHT)
        
        neighbors = self.grid.get_neighbors(5, 5)
        expected = [
            (3, 4), (3, 6), (4, 3), (4, 7),
            (6, 3), (6, 7), (7, 4), (7, 6)
        ]
        self.assertEqual(set(neighbors), set(expected))
        self.assertEqual(len(neighbors), 8)
        
        # Test knight movement near edge
        edge_neighbors = self.grid.get_neighbors(1, 1)
        # Should have fewer neighbors near edge
        self.assertLess(len(edge_neighbors), 8)
    
    def test_hex_movement_neighbors(self):
        """Test hexagonal movement neighbors."""
        self.grid.set_movement_type(MovementType.HEX)
        
        neighbors = self.grid.get_neighbors(5, 5)
        # Hex movement should have 6 neighbors for interior points
        self.assertEqual(len(neighbors), 6)
        
        # Check specific hex pattern (depends on row parity)
        # For odd row (5), expect specific pattern
        expected_directions = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)]
        expected = [(5 + dx, 5 + dy) for dx, dy in expected_directions]
        expected = [(x, y) for x, y in expected if 0 <= x < 10 and 0 <= y < 10]
        self.assertEqual(set(neighbors), set(expected))
    
    def test_custom_movement_neighbors(self):
        """Test custom movement pattern neighbors."""
        custom_moves = [(0, 3), (3, 0), (0, -3), (-3, 0)]  # Large cross pattern
        self.grid.set_custom_movement_pattern(custom_moves)
        
        neighbors = self.grid.get_neighbors(5, 5)
        expected = [(5, 8), (8, 5), (5, 2), (2, 5)]
        self.assertEqual(set(neighbors), set(expected))
    
    def test_terrain_cost_setting(self):
        """Test setting terrain costs."""
        # Set cost for a specific cell
        self.grid.set_terrain_cost(3, 4, 2.5)
        self.assertEqual(self.grid.get_terrain_cost(3, 4), 2.5)
        
        # Set cost for multiple cells
        costs = {(1, 1): 1.5, (2, 2): 3.0, (5, 5): 0.5}
        for (x, y), cost in costs.items():
            self.grid.set_terrain_cost(x, y, cost)
        
        for (x, y), expected_cost in costs.items():
            self.assertEqual(self.grid.get_terrain_cost(x, y), expected_cost)
    
    def test_movement_cost_calculation(self):
        """Test movement cost calculation."""
        # Set different terrain costs
        self.grid.set_terrain_cost(5, 5, 2.0)  # Source
        self.grid.set_terrain_cost(5, 6, 3.0)  # Destination
        
        # Test 4-connected movement cost
        self.grid.set_movement_type(MovementType.FOUR_CONNECTED)
        cost = self.grid.get_movement_cost(5, 5, 5, 6)
        expected_cost = (2.0 + 3.0) / 2  # Average of source and destination
        self.assertEqual(cost, expected_cost)
        
        # Test 8-connected diagonal movement cost
        self.grid.set_movement_type(MovementType.EIGHT_CONNECTED)
        diagonal_cost = self.grid.get_movement_cost(5, 5, 6, 6)
        expected_diagonal = ((2.0 + 1.0) / 2) * np.sqrt(2)  # Diagonal distance
        self.assertAlmostEqual(diagonal_cost, expected_diagonal, places=5)
    
    def test_bounds_checking(self):
        """Test that neighbors respect grid boundaries."""
        movement_types = [
            MovementType.FOUR_CONNECTED,
            MovementType.EIGHT_CONNECTED,
            MovementType.KNIGHT
        ]
        
        for movement_type in movement_types:
            with self.subTest(movement_type=movement_type):
                self.grid.set_movement_type(movement_type)
                
                # Test all corners and edges
                test_positions = [(0, 0), (0, 9), (9, 0), (9, 9), (0, 5), (9, 5), (5, 0), (5, 9)]
                
                for x, y in test_positions:
                    neighbors = self.grid.get_neighbors(x, y)
                    for nx, ny in neighbors:
                        self.assertGreaterEqual(nx, 0)
                        self.assertLess(nx, self.grid.width)
                        self.assertGreaterEqual(ny, 0)
                        self.assertLess(ny, self.grid.height)
    
    def test_obstacle_inheritance(self):
        """Test that obstacle functionality is inherited from base class."""
        # Add obstacles
        self.grid.add_obstacle(3, 3)
        self.grid.add_obstacle(4, 4)
        
        # Test obstacle checking
        self.assertTrue(self.grid.is_obstacle(3, 3))
        self.assertTrue(self.grid.is_obstacle(4, 4))
        self.assertFalse(self.grid.is_obstacle(5, 5))
        
        # Remove obstacle
        self.grid.remove_obstacle(3, 3)
        self.assertFalse(self.grid.is_obstacle(3, 3))
    
    def test_neighbor_filtering_with_obstacles(self):
        """Test that neighbors don't include obstacles."""
        # Add obstacle
        self.grid.add_obstacle(5, 6)
        
        # Get neighbors of adjacent cell
        neighbors = self.grid.get_neighbors(5, 5)
        
        # Obstacle should not be in neighbors
        self.assertNotIn((5, 6), neighbors)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for creating enhanced grids."""
    
    def test_create_grid_with_movement(self):
        """Test factory function for creating grids with specific movement."""
        grid = create_grid_with_movement(15, 20, MovementType.EIGHT_CONNECTED)
        
        self.assertEqual(grid.width, 15)
        self.assertEqual(grid.height, 20)
        self.assertEqual(grid.movement_type, MovementType.EIGHT_CONNECTED)
        self.assertIsInstance(grid, EnhancedGridMap)
    
    def test_create_terrain_grid(self):
        """Test factory function for creating terrain grids."""
        # Create terrain pattern
        terrain_costs = np.random.uniform(0.5, 3.0, (10, 15))
        
        grid = create_terrain_grid(terrain_costs, MovementType.HEX)
        
        self.assertEqual(grid.width, 10)
        self.assertEqual(grid.height, 15)
        self.assertEqual(grid.movement_type, MovementType.HEX)
        np.testing.assert_array_equal(grid.terrain_costs, terrain_costs)
    
    def test_create_terrain_grid_with_list(self):
        """Test terrain grid creation with list input."""
        terrain_list = [[1.0, 2.0, 1.5], [2.5, 1.0, 3.0]]
        
        grid = create_terrain_grid(terrain_list, MovementType.FOUR_CONNECTED)
        
        self.assertEqual(grid.width, 2)
        self.assertEqual(grid.height, 3)
        expected_array = np.array(terrain_list)
        np.testing.assert_array_equal(grid.terrain_costs, expected_array)


class TestMovementIntegration(unittest.TestCase):
    """Test integration of movement models with path planning."""
    
    def setUp(self):
        """Set up test environment."""
        self.grid = EnhancedGridMap(8, 8)
        # Add some obstacles to make pathfinding interesting
        for i in range(2, 6):
            self.grid.add_obstacle(i, 4)
    
    def test_movement_affects_pathfinding(self):
        """Test that different movement types affect available paths."""
        start = (0, 0)
        goal = (7, 7)
        
        # Test with 4-connected movement
        self.grid.set_movement_type(MovementType.FOUR_CONNECTED)
        neighbors_4 = self.grid.get_neighbors(1, 1)
        
        # Test with 8-connected movement
        self.grid.set_movement_type(MovementType.EIGHT_CONNECTED)
        neighbors_8 = self.grid.get_neighbors(1, 1)
        
        # 8-connected should have more neighbors
        self.assertGreater(len(neighbors_8), len(neighbors_4))
    
    def test_terrain_costs_affect_movement(self):
        """Test that terrain costs affect movement calculations."""
        # Create varied terrain
        self.grid.set_terrain_cost(2, 2, 5.0)  # High cost area
        self.grid.set_terrain_cost(3, 3, 0.1)  # Low cost area
        
        # Movement to high cost area should be expensive
        high_cost = self.grid.get_movement_cost(1, 1, 2, 2)
        
        # Movement to low cost area should be cheap
        low_cost = self.grid.get_movement_cost(1, 1, 3, 3)
        
        # Due to diagonal movement, exact comparison is complex,
        # but high cost should generally be more expensive
        self.assertGreater(high_cost, 1.0)
        self.assertLess(low_cost, 2.0)
    
    def test_knight_movement_unique_paths(self):
        """Test that knight movement provides unique connectivity."""
        self.grid.set_movement_type(MovementType.KNIGHT)
        
        # Knight can jump over obstacles
        # Place obstacles around a position
        center = (4, 4)
        surrounding_obstacles = [
            (3, 3), (3, 4), (3, 5),
            (4, 3),         (4, 5),
            (5, 3), (5, 4), (5, 5)
        ]
        
        for x, y in surrounding_obstacles:
            if 0 <= x < 8 and 0 <= y < 8:
                self.grid.add_obstacle(x, y)
        
        # Knight should still be able to move from center
        neighbors = self.grid.get_neighbors(4, 4)
        self.assertGreater(len(neighbors), 0)  # Should have some valid moves
        
        # Verify that knight moves jump over obstacles
        for nx, ny in neighbors:
            # Knight moves should not be adjacent to center
            self.assertGreater(abs(nx - 4) + abs(ny - 4), 2)


if __name__ == '__main__':
    unittest.main()
