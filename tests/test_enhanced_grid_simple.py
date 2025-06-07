"""
Simplified test suite for enhanced grid that matches our actual implementation.
"""

import unittest
import sys
import os

# Add the parent directory to the path to import amr_path_planner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_path_planner.grid_map import GridMap
from amr_path_planner.enhanced_grid import EnhancedGridMap


class TestEnhancedGridMap(unittest.TestCase):
    """Test the EnhancedGridMap class."""
    
    def setUp(self):
        """Set up test environment."""
        self.width = 10
        self.height = 10
        self.obstacles = {(3, 3), (3, 4), (4, 3)}
    
    def test_initialization(self):
        """Test enhanced grid initialization."""
        # Default 4-connected
        grid = EnhancedGridMap(self.width, self.height, self.obstacles)
        self.assertEqual(grid.width, self.width)
        self.assertEqual(grid.height, self.height)
        self.assertEqual(grid.movement_model, '4-connected')
        
        # 8-connected
        grid8 = EnhancedGridMap(self.width, self.height, movement_model='8-connected')
        self.assertEqual(grid8.movement_model, '8-connected')
        
        # Custom movement
        custom_moves = [(1, 0), (0, 1), (-1, 0), (0, -1), (2, 1)]
        grid_custom = EnhancedGridMap(self.width, self.height, 
                                    movement_model='custom', 
                                    custom_moves=custom_moves)
        self.assertEqual(grid_custom.movement_model, 'custom')
        self.assertEqual(grid_custom.custom_moves, custom_moves)
    
    def test_invalid_movement_model(self):
        """Test invalid movement model."""
        with self.assertRaises(ValueError):
            EnhancedGridMap(10, 10, movement_model='invalid')
        
        # Custom without moves
        with self.assertRaises(ValueError):
            EnhancedGridMap(10, 10, movement_model='custom')
    
    def test_neighbors_4_connected(self):
        """Test 4-connected movement."""
        grid = EnhancedGridMap(10, 10, movement_model='4-connected')
        
        # Center position
        neighbors = grid.neighbors(5, 5)
        expected = [(6, 5), (4, 5), (5, 6), (5, 4)]
        self.assertEqual(set(neighbors), set(expected))
        
        # Corner position
        neighbors = grid.neighbors(0, 0)
        expected = [(1, 0), (0, 1)]
        self.assertEqual(set(neighbors), set(expected))
    
    def test_neighbors_8_connected(self):
        """Test 8-connected movement."""
        grid = EnhancedGridMap(10, 10, movement_model='8-connected')
        
        # Center position should have 8 neighbors
        neighbors = grid.neighbors(5, 5)
        self.assertEqual(len(neighbors), 8)
        
        # Should include diagonal moves
        diagonal_neighbors = [(4, 4), (4, 6), (6, 4), (6, 6)]
        for neighbor in diagonal_neighbors:
            self.assertIn(neighbor, neighbors)
    
    def test_neighbors_custom_movement(self):
        """Test custom movement patterns."""
        # Knight-like movement
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), 
                       (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        grid = EnhancedGridMap(10, 10, movement_model='custom', 
                             custom_moves=knight_moves)
        
        neighbors = grid.neighbors(5, 5)
        
        # Should only include valid positions within bounds
        expected_neighbors = []
        for dx, dy in knight_moves:
            new_x, new_y = 5 + dx, 5 + dy
            if 0 <= new_x < 10 and 0 <= new_y < 10:
                expected_neighbors.append((new_x, new_y))
        
        self.assertEqual(set(neighbors), set(expected_neighbors))
    
    def test_neighbors_with_obstacles(self):
        """Test that neighbors excludes obstacles."""
        obstacles = {(6, 5), (4, 5)}
        grid = EnhancedGridMap(10, 10, static_obstacles=obstacles, 
                             movement_model='4-connected')
        
        neighbors = grid.neighbors(5, 5)
        
        # Should not include obstacle positions
        for obstacle in obstacles:
            self.assertNotIn(obstacle, neighbors)
    
    def test_cost_function(self):
        """Test basic cost functionality."""
        grid = EnhancedGridMap(10, 10)
        
        # Test setting and getting costs
        grid.cost_map[(5, 5)] = 2.5
        
        # Should have the custom cost
        self.assertEqual(grid.cost_map.get((5, 5), 1.0), 2.5)
        
        # Default cost should be 1.0
        self.assertEqual(grid.cost_map.get((3, 3), 1.0), 1.0)
    
    def test_inheritance_from_gridmap(self):
        """Test that EnhancedGridMap properly inherits from GridMap."""
        grid = EnhancedGridMap(10, 10, self.obstacles)
          # Should inherit basic GridMap functionality
        self.assertFalse(grid.is_free(3, 3))  # Obstacle position
        self.assertTrue(grid.is_free(5, 5))   # Free space
        
        # Should work with bounds checking
        self.assertFalse(grid.is_free(15, 15))  # Out of bounds
        self.assertTrue(grid.is_free(5, 5))     # Free space
        
        # Should allow adding/removing obstacles
        grid.add_obstacle(7, 7)
        self.assertFalse(grid.is_free(7, 7))
        
        grid.remove_obstacle(7, 7)
        self.assertTrue(grid.is_free(7, 7))
    
    def test_movement_pattern_factory_methods(self):
        """Test that we can create grids with different movement patterns easily."""
        # Test different patterns work
        patterns = ['4-connected', '8-connected']
        
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                grid = EnhancedGridMap(10, 10, movement_model=pattern)
                neighbors = grid.neighbors(5, 5)
                
                # Should return valid neighbors
                self.assertGreater(len(neighbors), 0)
                
                # All neighbors should be valid positions
                for x, y in neighbors:
                    self.assertTrue(0 <= x < 10)
                    self.assertTrue(0 <= y < 10)


if __name__ == '__main__':
    unittest.main()
