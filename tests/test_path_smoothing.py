"""
Test suite for path smoothing algorithms.
"""

import unittest
import numpy as np
import sys
import os
from unittest.mock import patch

# Add the parent directory to the path to import amr_path_planner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_path_planner.grid_map import GridMap
from amr_path_planner.path_smoothing import (
    shortcut_smoothing, bezier_smoothing, spline_smoothing,
    adaptive_smoothing, douglas_peucker_smoothing,
    path_length, path_curvature_analysis,  # Use actual function names
    euclidean_distance, line_collision_check, smooth_path
)


class TestPathSmoothingUtilities(unittest.TestCase):
    """Test utility functions for path analysis."""
    
    def test_calculate_path_length(self):
        """Test path length calculation."""
        # Simple straight line path
        path = [(0, 0), (3, 4), (6, 8)]
        length = calculate_path_length(path)
        expected = 5.0 + 5.0  # Two segments of length 5
        self.assertAlmostEqual(length, expected, places=5)
        
        # Single point path
        single_point = [(0, 0)]
        self.assertEqual(calculate_path_length(single_point), 0.0)
        
        # Empty path
        self.assertEqual(calculate_path_length([]), 0.0)
    
    def test_calculate_path_curvature(self):
        """Test path curvature calculation."""
        # Straight line (zero curvature)
        straight_path = [(0, 0), (1, 1), (2, 2), (3, 3)]
        curvatures = calculate_path_curvature(straight_path)
        for curvature in curvatures:
            self.assertAlmostEqual(curvature, 0.0, places=5)
        
        # Right angle turn (high curvature)
        right_angle = [(0, 0), (1, 0), (1, 1)]
        curvatures = calculate_path_curvature(right_angle)
        # Middle point should have high curvature
        self.assertGreater(abs(curvatures[0]), 0.5)
    
    def test_analyze_path_smoothness(self):
        """Test path smoothness analysis."""
        # Create a path with a sharp turn
        zigzag_path = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]
        
        analysis = analyze_path_smoothness(zigzag_path)
        
        self.assertIn('length', analysis)
        self.assertIn('max_curvature', analysis)
        self.assertIn('avg_curvature', analysis)
        self.assertIn('smoothness_score', analysis)
        
        self.assertGreater(analysis['length'], 0)
        self.assertGreaterEqual(analysis['max_curvature'], 0)
        self.assertGreaterEqual(analysis['avg_curvature'], 0)
        self.assertGreaterEqual(analysis['smoothness_score'], 0)


class TestShortcutSmoothing(unittest.TestCase):
    """Test shortcut smoothing algorithm."""
    
    def setUp(self):
        """Set up test environment."""
        self.grid = GridMap(10, 10)
        # Add some obstacles
        self.grid.add_obstacle(3, 3)
        self.grid.add_obstacle(3, 4)
        self.grid.add_obstacle(4, 3)
    
    def test_shortcut_smoothing_simple(self):
        """Test shortcut smoothing on a simple path."""
        # Create a path that goes around an obstacle unnecessarily
        original_path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]
        
        smoothed_path = shortcut_smoothing(original_path, self.grid, max_iterations=100)
        
        # Smoothed path should be shorter or equal
        original_length = calculate_path_length(original_path)
        smoothed_length = calculate_path_length(smoothed_path)
        self.assertLessEqual(smoothed_length, original_length)
        
        # Start and end should be the same
        self.assertEqual(smoothed_path[0], original_path[0])
        self.assertEqual(smoothed_path[-1], original_path[-1])
    
    def test_shortcut_smoothing_with_obstacles(self):
        """Test that shortcut smoothing respects obstacles."""
        # Create a path that must go around obstacles
        path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 5), (5, 5)]
        
        smoothed_path = shortcut_smoothing(path, self.grid)
        
        # Check that no point in smoothed path is an obstacle
        for x, y in smoothed_path:
            self.assertFalse(self.grid.is_obstacle(int(x), int(y)))
    
    def test_shortcut_smoothing_empty_path(self):
        """Test shortcut smoothing with edge cases."""
        # Empty path
        self.assertEqual(shortcut_smoothing([], self.grid), [])
        
        # Single point
        single_point = [(5, 5)]
        self.assertEqual(shortcut_smoothing(single_point, self.grid), single_point)
        
        # Two points
        two_points = [(0, 0), (1, 1)]
        result = shortcut_smoothing(two_points, self.grid)
        self.assertEqual(len(result), 2)


class TestBezierSmoothing(unittest.TestCase):
    """Test Bezier curve smoothing."""
    
    def test_bezier_smoothing_basic(self):
        """Test basic Bezier smoothing functionality."""
        # Simple path
        path = [(0, 0), (2, 2), (4, 2), (6, 4)]
        
        smoothed_path = bezier_smoothing(path, num_points=20)
        
        # Should have more points than original
        self.assertGreaterEqual(len(smoothed_path), len(path))
        
        # First and last points should be preserved
        self.assertEqual(smoothed_path[0], path[0])
        self.assertEqual(smoothed_path[-1], path[-1])
    
    def test_bezier_smoothing_short_path(self):
        """Test Bezier smoothing on short paths."""
        # Path with only 2 points
        short_path = [(0, 0), (1, 1)]
        smoothed = bezier_smoothing(short_path)
        self.assertGreaterEqual(len(smoothed), 2)
        
        # Single point
        single_point = [(5, 5)]
        self.assertEqual(bezier_smoothing(single_point), single_point)
        
        # Empty path
        self.assertEqual(bezier_smoothing([]), [])


class TestSplineSmoothing(unittest.TestCase):
    """Test spline smoothing (requires scipy)."""
    
    def test_spline_smoothing_basic(self):
        """Test basic spline smoothing functionality."""
        path = [(0, 0), (1, 2), (3, 1), (4, 3), (6, 2)]
        
        try:
            smoothed_path = spline_smoothing(path, num_points=20)
            
            # Should have specified number of points
            self.assertEqual(len(smoothed_path), 20)
            
            # First and last points should be close to original
            self.assertAlmostEqual(smoothed_path[0][0], path[0][0], places=1)
            self.assertAlmostEqual(smoothed_path[0][1], path[0][1], places=1)
            self.assertAlmostEqual(smoothed_path[-1][0], path[-1][0], places=1)
            self.assertAlmostEqual(smoothed_path[-1][1], path[-1][1], places=1)
            
        except ImportError:
            # Skip test if scipy is not available
            self.skipTest("scipy not available for spline smoothing")
    
    def test_spline_smoothing_edge_cases(self):
        """Test spline smoothing edge cases."""
        try:
            # Short path
            short_path = [(0, 0), (1, 1)]
            smoothed = spline_smoothing(short_path)
            self.assertGreaterEqual(len(smoothed), 2)
            
            # Single point returns as-is
            single = [(5, 5)]
            self.assertEqual(spline_smoothing(single), single)
            
            # Empty path
            self.assertEqual(spline_smoothing([]), [])
            
        except ImportError:
            self.skipTest("scipy not available for spline smoothing")


class TestAdaptiveSmoothing(unittest.TestCase):
    """Test adaptive smoothing based on curvature."""
    
    def test_adaptive_smoothing_basic(self):
        """Test basic adaptive smoothing."""
        # Create a path with varying curvature
        path = [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (4, 2), (5, 2)]
        
        smoothed_path = adaptive_smoothing(path, curvature_threshold=1.0)
        
        # Should return a valid path
        self.assertGreaterEqual(len(smoothed_path), 2)
        self.assertEqual(smoothed_path[0], path[0])
        self.assertEqual(smoothed_path[-1], path[-1])
    
    def test_adaptive_smoothing_straight_line(self):
        """Test adaptive smoothing on straight line (low curvature)."""
        straight_path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        
        smoothed_path = adaptive_smoothing(straight_path, curvature_threshold=0.5)
        
        # Straight line should not need much smoothing
        self.assertLessEqual(len(smoothed_path), len(straight_path) + 2)
    
    def test_adaptive_smoothing_edge_cases(self):
        """Test adaptive smoothing edge cases."""
        # Empty path
        self.assertEqual(adaptive_smoothing([]), [])
        
        # Single point
        single = [(5, 5)]
        self.assertEqual(adaptive_smoothing(single), single)
        
        # Two points
        two_points = [(0, 0), (1, 1)]
        result = adaptive_smoothing(two_points)
        self.assertGreaterEqual(len(result), 2)


class TestDouglasPeuckerSmoothing(unittest.TestCase):
    """Test Douglas-Peucker path simplification."""
    
    def test_douglas_peucker_basic(self):
        """Test basic Douglas-Peucker simplification."""
        # Create a path with redundant points
        path = [(0, 0), (1, 0.1), (2, 0), (3, 0.1), (4, 0), (5, 0)]
        
        simplified = douglas_peucker_smoothing(path, epsilon=0.2)
        
        # Should remove some redundant points
        self.assertLessEqual(len(simplified), len(path))
        
        # First and last points preserved
        self.assertEqual(simplified[0], path[0])
        self.assertEqual(simplified[-1], path[-1])
    
    def test_douglas_peucker_exact_line(self):
        """Test Douglas-Peucker on exact straight line."""
        # Perfect straight line
        straight_line = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        
        simplified = douglas_peucker_smoothing(straight_line, epsilon=0.1)
        
        # Should reduce to just start and end points
        self.assertEqual(len(simplified), 2)
        self.assertEqual(simplified[0], straight_line[0])
        self.assertEqual(simplified[-1], straight_line[-1])
    
    def test_douglas_peucker_edge_cases(self):
        """Test Douglas-Peucker edge cases."""
        # Empty path
        self.assertEqual(douglas_peucker_smoothing([]), [])
        
        # Single point
        single = [(5, 5)]
        self.assertEqual(douglas_peucker_smoothing(single), single)
        
        # Two points
        two_points = [(0, 0), (1, 1)]
        self.assertEqual(douglas_peucker_smoothing(two_points), two_points)


class TestSmoothingComparison(unittest.TestCase):
    """Test comparison between different smoothing methods."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a zigzag path that could benefit from smoothing
        self.zigzag_path = [
            (0, 0), (1, 0), (1, 1), (2, 1), (2, 2),
            (3, 2), (3, 3), (4, 3), (4, 4), (5, 4)
        ]
        
        self.grid = GridMap(10, 10)
    
    def test_smoothing_methods_comparison(self):
        """Test that different smoothing methods produce valid results."""
        methods = {
            'shortcut': lambda p: shortcut_smoothing(p, self.grid, max_iterations=50),
            'bezier': lambda p: bezier_smoothing(p, num_points=15),
            'adaptive': lambda p: adaptive_smoothing(p, curvature_threshold=1.0),
            'douglas_peucker': lambda p: douglas_peucker_smoothing(p, epsilon=0.5)
        }
        
        original_analysis = analyze_path_smoothness(self.zigzag_path)
        
        for method_name, method_func in methods.items():
            with self.subTest(method=method_name):
                try:
                    smoothed_path = method_func(self.zigzag_path)
                    
                    # Basic validations
                    self.assertGreaterEqual(len(smoothed_path), 2)
                    self.assertEqual(smoothed_path[0], self.zigzag_path[0])
                    self.assertEqual(smoothed_path[-1], self.zigzag_path[-1])
                    
                    # Analyze smoothness
                    smoothed_analysis = analyze_path_smoothness(smoothed_path)
                    
                    # Smoothed path should generally have better smoothness score
                    # (Note: This is not guaranteed for all methods, so we just check validity)
                    self.assertGreaterEqual(smoothed_analysis['smoothness_score'], 0)
                    
                except ImportError:
                    # Skip if dependencies not available
                    self.skipTest(f"Dependencies not available for {method_name}")
    
    def test_smoothing_preserves_endpoints(self):
        """Test that all smoothing methods preserve start and end points."""
        path = [(0, 0), (2, 1), (3, 3), (5, 2), (7, 4)]
        
        methods = [
            shortcut_smoothing,
            bezier_smoothing,
            adaptive_smoothing,
            douglas_peucker_smoothing
        ]
        
        for method in methods:
            with self.subTest(method=method.__name__):
                try:
                    if method == shortcut_smoothing:
                        smoothed = method(path, self.grid)
                    else:
                        smoothed = method(path)
                    
                    if len(smoothed) > 0:
                        self.assertEqual(smoothed[0], path[0])
                        self.assertEqual(smoothed[-1], path[-1])
                        
                except ImportError:
                    self.skipTest(f"Dependencies not available for {method.__name__}")


if __name__ == '__main__':
    unittest.main()
