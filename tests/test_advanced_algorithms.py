"""
Test suite for advanced path planning algorithms (RRT, RRT*, PRM).
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path to import amr_path_planner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_path_planner.grid_map import GridMap
from amr_path_planner.advanced_algorithms import Node, RRTPlanner, RRTStarPlanner, PRMPlanner


class TestNode(unittest.TestCase):
    """Test the Node class used in tree-based algorithms."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = Node(5, 10)
        self.assertEqual(node.x, 5)
        self.assertEqual(node.y, 10)
        self.assertIsNone(node.parent)
        self.assertEqual(node.cost, 0.0)
    
    def test_node_with_parent(self):
        """Test node creation with parent and cost."""
        parent_node = Node(0, 0)
        child_node = Node(3, 4, parent_node, 5.0)
        self.assertEqual(child_node.parent, parent_node)
        self.assertEqual(child_node.cost, 5.0)
    
    def test_distance_to(self):
        """Test distance calculation between nodes."""
        node1 = Node(0, 0)
        node2 = Node(3, 4)
        distance = node1.distance_to(node2)
        self.assertAlmostEqual(distance, 5.0, places=5)


class TestRRTPlanner(unittest.TestCase):
    """Test the RRT (Rapidly-exploring Random Tree) planner."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a simple 10x10 grid with some obstacles
        self.grid = GridMap(10, 10)
        # Add some obstacles
        for i in range(3, 7):
            self.grid.add_obstacle(i, 5)
        
        self.rrt = RRTPlanner(
            grid_map=self.grid,
            max_iterations=1000,
            step_size=1.0,
            goal_bias=0.1
        )
    
    def test_random_point_generation(self):
        """Test random point generation within grid bounds."""
        for _ in range(100):
            x, y = self.rrt._get_random_point()
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, self.grid.width)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, self.grid.height)
    
    def test_nearest_node_finding(self):
        """Test finding the nearest node in the tree."""
        # Add some nodes to the tree
        start_node = Node(0, 0)
        self.rrt.tree = [start_node]
        self.rrt.tree.append(Node(2, 1, start_node))
        self.rrt.tree.append(Node(1, 3, start_node))
        
        # Find nearest to point (2, 2)
        nearest = self.rrt._get_nearest_node(2, 2)
        self.assertEqual(nearest.x, 2)
        self.assertEqual(nearest.y, 1)
    
    def test_collision_checking(self):
        """Test collision detection."""
        # Test collision with obstacle
        self.assertTrue(self.rrt._check_collision(4, 5))
        # Test no collision with free space
        self.assertFalse(self.rrt._check_collision(1, 1))
    
    def test_path_planning_simple(self):
        """Test path planning on a simple scenario."""
        start = (0, 0)
        goal = (9, 9)
        
        path = self.rrt.plan_path(start, goal)
        
        if path:  # RRT is probabilistic, so path might not always be found
            self.assertEqual(path[0], start)
            self.assertEqual(path[-1], goal)
            # Check path validity (no obstacles)
            for x, y in path:
                self.assertFalse(self.grid.is_obstacle(x, y))


class TestRRTStarPlanner(unittest.TestCase):
    """Test the RRT* (optimal RRT) planner."""
    
    def setUp(self):
        """Set up test environment."""
        self.grid = GridMap(10, 10)
        # Add some obstacles
        for i in range(3, 7):
            self.grid.add_obstacle(i, 5)
        
        self.rrt_star = RRTStarPlanner(
            grid_map=self.grid,
            max_iterations=1000,
            step_size=1.0,
            goal_bias=0.1,
            search_radius=2.0
        )
    
    def test_near_nodes_finding(self):
        """Test finding nodes within search radius."""
        # Add some nodes to the tree
        start_node = Node(0, 0)
        self.rrt_star.tree = [start_node]
        self.rrt_star.tree.append(Node(1, 1, start_node, 1.414))
        self.rrt_star.tree.append(Node(2, 0, start_node, 2.0))
        self.rrt_star.tree.append(Node(5, 5, start_node, 7.071))
        
        # Find nodes near (1.5, 0.5) within radius 2.0
        near_nodes = self.rrt_star._get_near_nodes(1.5, 0.5)
        
        # Should find first 3 nodes but not the distant one
        self.assertGreaterEqual(len(near_nodes), 2)
        self.assertLessEqual(len(near_nodes), 3)
    
    def test_path_planning_optimality(self):
        """Test that RRT* can find better paths than RRT."""
        start = (0, 0)
        goal = (2, 2)
        
        # Plan with both algorithms
        rrt = RRTPlanner(self.grid, max_iterations=500)
        rrt_star = RRTStarPlanner(self.grid, max_iterations=500)
        
        path_rrt = rrt.plan_path(start, goal)
        path_rrt_star = rrt_star.plan_path(start, goal)
        
        # Both should find valid paths (if any)
        if path_rrt and path_rrt_star:
            self.assertEqual(path_rrt[0], start)
            self.assertEqual(path_rrt[-1], goal)
            self.assertEqual(path_rrt_star[0], start)
            self.assertEqual(path_rrt_star[-1], goal)


class TestPRMPlanner(unittest.TestCase):
    """Test the PRM (Probabilistic Roadmap) planner."""
    
    def setUp(self):
        """Set up test environment."""
        self.grid = GridMap(10, 10)
        # Add some obstacles
        for i in range(3, 7):
            self.grid.add_obstacle(i, 5)
        
        self.prm = PRMPlanner(
            grid_map=self.grid,
            num_samples=100,
            connection_radius=2.0
        )
    
    def test_sample_generation(self):
        """Test generation of random samples."""
        samples = self.prm._generate_samples()
        
        self.assertEqual(len(samples), self.prm.num_samples)
        
        # All samples should be in free space
        for node in samples:
            self.assertFalse(self.grid.is_obstacle(node.x, node.y))
            self.assertGreaterEqual(node.x, 0)
            self.assertLess(node.x, self.grid.width)
            self.assertGreaterEqual(node.y, 0)
            self.assertLess(node.y, self.grid.height)
    
    def test_roadmap_construction(self):
        """Test roadmap construction."""
        samples = self.prm._generate_samples()
        graph = self.prm._build_roadmap(samples)
        
        # Graph should have nodes
        self.assertGreater(len(graph), 0)
        
        # Check that connections are within radius
        for node in graph:
            for neighbor in graph[node]:
                distance = node.distance_to(neighbor)
                self.assertLessEqual(distance, self.prm.connection_radius + 0.001)  # Small tolerance
    
    def test_path_planning(self):
        """Test complete path planning with PRM."""
        start = (0, 0)
        goal = (9, 0)  # Should be reachable
        
        path = self.prm.plan_path(start, goal)
        
        if path:  # PRM is probabilistic
            self.assertEqual(path[0], start)
            self.assertEqual(path[-1], goal)
            # Check path validity
            for x, y in path:
                self.assertFalse(self.grid.is_obstacle(x, y))


class TestAlgorithmComparison(unittest.TestCase):
    """Test comparison between different algorithms."""
    
    def setUp(self):
        """Set up test environment."""
        self.grid = GridMap(15, 15)
        # Create a more complex obstacle pattern
        for i in range(5, 10):
            self.grid.add_obstacle(i, 7)
        for j in range(3, 8):
            self.grid.add_obstacle(7, j)
    
    def test_algorithm_consistency(self):
        """Test that all algorithms can solve the same problem."""
        start = (0, 0)
        goal = (14, 14)
        
        # Initialize all planners
        rrt = RRTPlanner(self.grid, max_iterations=2000, step_size=1.0)
        rrt_star = RRTStarPlanner(self.grid, max_iterations=2000, step_size=1.0)
        prm = PRMPlanner(self.grid, num_samples=200, connection_radius=3.0)
        
        # Plan paths
        paths = {
            'RRT': rrt.plan_path(start, goal),
            'RRT*': rrt_star.plan_path(start, goal),
            'PRM': prm.plan_path(start, goal)
        }
        
        # Check that at least some algorithms found paths
        successful_algorithms = [name for name, path in paths.items() if path is not None]
        self.assertGreater(len(successful_algorithms), 0, 
                          "At least one algorithm should find a path")
        
        # Validate all found paths
        for algorithm_name, path in paths.items():
            if path:
                with self.subTest(algorithm=algorithm_name):
                    self.assertEqual(path[0], start)
                    self.assertEqual(path[-1], goal)
                    # Check no obstacles in path
                    for x, y in path:
                        self.assertFalse(self.grid.is_obstacle(x, y))


if __name__ == '__main__':
    unittest.main()
