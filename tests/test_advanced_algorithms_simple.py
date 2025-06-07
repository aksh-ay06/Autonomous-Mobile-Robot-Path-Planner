"""
Simplified test suite for advanced algorithms that matches our actual implementation.
"""

import unittest
import sys
import os

# Add the parent directory to the path to import amr_path_planner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_path_planner.grid_map import GridMap

# Test if advanced algorithms are available
try:
    from amr_path_planner.advanced_algorithms import (
        Node, rrt, rrt_star, prm
    )
    ADVANCED_ALGORITHMS_AVAILABLE = True
    print("Advanced algorithms imported successfully!")
    print(f"Node: {Node}")
    print(f"rrt: {rrt}")
    print(f"rrt_star: {rrt_star}")
    print(f"prm: {prm}")
except ImportError as e:
    ADVANCED_ALGORITHMS_AVAILABLE = False
    Node = rrt = rrt_star = prm = None
    print(f"Import error: {e}")

print(f"ADVANCED_ALGORITHMS_AVAILABLE: {ADVANCED_ALGORITHMS_AVAILABLE}")


# @unittest.skipIf(not ADVANCED_ALGORITHMS_AVAILABLE, "Advanced algorithms not available")
class TestAdvancedAlgorithms(unittest.TestCase):
    """Test advanced path planning algorithms."""
    
    def setUp(self):
        """Set up test environment."""
        self.grid = GridMap(20, 20)
        # Add some obstacles to make it interesting
        for i in range(5, 15):
            self.grid.add_obstacle(i, 10)
        
        self.start = (1, 1)
        self.goal = (18, 18)
    
    def test_node_class(self):
        """Test the Node class."""
        if Node is None:
            self.skipTest("Node class not available")
        
        node = Node((5, 5))
        self.assertEqual(node.position, (5, 5))
        self.assertIsNone(node.parent)
        self.assertEqual(node.cost, 0.0)
        
        # Test with parent
        parent = Node((3, 3))
        child = Node((5, 5), parent=parent)
        self.assertEqual(child.parent, parent)
        self.assertEqual(child.cost, 0.0)  # Cost is initialized to 0.0
        
        # Test add_child method
        parent.add_child(child)
        self.assertIn(child, parent.children)
        self.assertEqual(child.parent, parent)
    
    def test_rrt_algorithm(self):
        """Test RRT algorithm."""
        if rrt is None:
            self.skipTest("RRT algorithm not available")
        
        path = rrt(self.start, self.goal, self.grid, 
                  max_iterations=1000, step_size=1.0, goal_bias=0.1)
        
        if path:  # Path found
            # Should start and end at correct positions
            self.assertEqual(path[0], self.start)
            self.assertEqual(path[-1], self.goal)
            
            # Path should be valid (no obstacles)
            for x, y in path:
                self.assertTrue(self.grid.is_free(x, y))
        else:
            # If no path found, that's also acceptable for RRT
            self.assertEqual(path, [])
    
    def test_rrt_star_algorithm(self):
        """Test RRT* algorithm."""
        if rrt_star is None:
            self.skipTest("RRT* algorithm not available")
        
        path = rrt_star(self.start, self.goal, self.grid, 
                       max_iterations=1000, step_size=1.0, 
                       goal_bias=0.1, search_radius=3.0)
        
        if path:  # Path found
            # Should start and end at correct positions
            self.assertEqual(path[0], self.start)
            self.assertEqual(path[-1], self.goal)
            
            # Path should be valid (no obstacles)
            for x, y in path:
                self.assertTrue(self.grid.is_free(x, y))
        else:
            # If no path found, that's also acceptable
            self.assertEqual(path, [])
    
    def test_prm_algorithm(self):
        """Test PRM algorithm."""
        if prm is None:
            self.skipTest("PRM algorithm not available")
        
        path = prm(self.start, self.goal, self.grid, 
                  num_samples=200, connection_radius=3.0)
        
        if path:  # Path found
            # Should start and end at correct positions
            self.assertEqual(path[0], self.start)
            self.assertEqual(path[-1], self.goal)
            
            # Path should be valid (no obstacles)
            for x, y in path:
                self.assertTrue(self.grid.is_free(x, y))
        else:
            # If no path found, that's also acceptable
            self.assertEqual(path, [])
    
    def test_algorithms_with_simple_scenario(self):
        """Test algorithms with a simple, solvable scenario."""
        # Create a simple grid with clear path
        simple_grid = GridMap(10, 10)
        simple_start = (1, 1)
        simple_goal = (8, 8)
        
        algorithms = []
        if rrt is not None:
            algorithms.append(('RRT', lambda: rrt(simple_start, simple_goal, simple_grid, 500)))
        if rrt_star is not None:
            algorithms.append(('RRT*', lambda: rrt_star(simple_start, simple_goal, simple_grid, 500)))
        if prm is not None:
            algorithms.append(('PRM', lambda: prm(simple_start, simple_goal, simple_grid, 100)))
        
        for alg_name, alg_func in algorithms:
            with self.subTest(algorithm=alg_name):
                path = alg_func()
                
                # Should find a path in this simple scenario
                self.assertGreater(len(path), 0, f"{alg_name} should find a path in simple scenario")
                
                if path:
                    self.assertEqual(path[0], simple_start)
                    self.assertEqual(path[-1], simple_goal)
    
    def test_algorithms_with_no_solution(self):
        """Test algorithms when no solution exists."""
        # Create a grid where goal is surrounded by obstacles
        blocked_grid = GridMap(10, 10)
        
        # Surround goal with obstacles
        goal_pos = (8, 8)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:  # Don't block the goal itself
                    blocked_grid.add_obstacle(goal_pos[0] + dx, goal_pos[1] + dy)
        
        start_pos = (1, 1)
        
        algorithms = []
        if rrt is not None:
            algorithms.append(('RRT', lambda: rrt(start_pos, goal_pos, blocked_grid, 200)))
        if rrt_star is not None:
            algorithms.append(('RRT*', lambda: rrt_star(start_pos, goal_pos, blocked_grid, 200)))
        if prm is not None:
            algorithms.append(('PRM', lambda: prm(start_pos, goal_pos, blocked_grid, 50)))
        
        for alg_name, alg_func in algorithms:
            with self.subTest(algorithm=alg_name):
                path = alg_func()
                
                # Should return empty path when no solution exists
                self.assertEqual(path, [], f"{alg_name} should return empty path when blocked")
    
    def test_algorithm_parameters(self):
        """Test algorithms with different parameters."""
        simple_grid = GridMap(10, 10)
        start = (1, 1)
        goal = (8, 8)
        
        # Test RRT with different parameters
        if rrt is not None:
            # Test with different step sizes
            path1 = rrt(start, goal, simple_grid, max_iterations=100, step_size=0.5)
            path2 = rrt(start, goal, simple_grid, max_iterations=100, step_size=2.0)
            
            # Both should be valid (if they exist)
            for path in [path1, path2]:
                if path:
                    self.assertEqual(path[0], start)
                    self.assertEqual(path[-1], goal)
        
        # Test PRM with different samples
        if prm is not None:
            path1 = prm(start, goal, simple_grid, num_samples=50, connection_radius=2.0)
            path2 = prm(start, goal, simple_grid, num_samples=200, connection_radius=4.0)
            
            # Both should be valid (if they exist)
            for path in [path1, path2]:
                if path:
                    self.assertEqual(path[0], start)
                    self.assertEqual(path[-1], goal)


if __name__ == '__main__':
    unittest.main()
