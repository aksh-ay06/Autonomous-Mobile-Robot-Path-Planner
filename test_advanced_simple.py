#!/usr/bin/env python3
"""
Simple test runner for advanced algorithms.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.abspath('.'))

from amr_path_planner.advanced_algorithms import Node, rrt, rrt_star, prm
from amr_path_planner.grid_map import GridMap

def test_node_class():
    """Test the Node class."""
    print("Testing Node class...")
    
    node = Node((5, 5))
    assert node.position == (5, 5), f"Expected (5, 5), got {node.position}"
    assert node.parent is None, f"Expected None, got {node.parent}"
    assert node.cost == 0.0, f"Expected 0.0, got {node.cost}"
    
    # Test with parent
    parent = Node((3, 3))
    child = Node((5, 5), parent=parent)
    assert child.parent == parent, f"Expected {parent}, got {child.parent}"
    assert child.cost == 0.0, f"Expected 0.0, got {child.cost}"
    
    # Test add_child method
    parent.add_child(child)
    assert child in parent.children, f"Child not found in parent's children"
    assert child.parent == parent, f"Expected {parent}, got {child.parent}"
    
    print("  ✓ Node class tests passed!")

def test_rrt_algorithm():
    """Test RRT algorithm."""
    print("Testing RRT algorithm...")
    
    grid = GridMap(20, 20)
    # Add some obstacles
    for i in range(5, 15):
        grid.add_obstacle(i, 10)
    
    start = (1, 1)
    goal = (18, 18)
    
    path = rrt(start, goal, grid, max_iterations=1000, step_size=1.0, goal_bias=0.1)
    
    if path:  # Path found
        assert path[0] == start, f"Expected start {start}, got {path[0]}"
        assert path[-1] == goal, f"Expected goal {goal}, got {path[-1]}"
        
        # Path should be valid (no obstacles)
        for x, y in path:
            assert grid.is_free(x, y), f"Path goes through obstacle at ({x}, {y})"
    else:
        # If no path found, that's also acceptable for RRT
        assert path == [], f"Expected empty list, got {path}"
    
    print("  ✓ RRT algorithm tests passed!")

def test_rrt_star_algorithm():
    """Test RRT* algorithm."""
    print("Testing RRT* algorithm...")
    
    grid = GridMap(20, 20)
    # Add some obstacles
    for i in range(5, 15):
        grid.add_obstacle(i, 10)
    
    start = (1, 1)
    goal = (18, 18)
    
    path = rrt_star(start, goal, grid, 
                   max_iterations=1000, step_size=1.0, 
                   goal_bias=0.1, search_radius=3.0)
    
    if path:  # Path found
        assert path[0] == start, f"Expected start {start}, got {path[0]}"
        assert path[-1] == goal, f"Expected goal {goal}, got {path[-1]}"
        
        # Path should be valid (no obstacles)
        for x, y in path:
            assert grid.is_free(x, y), f"Path goes through obstacle at ({x}, {y})"
    else:
        # If no path found, that's also acceptable
        assert path == [], f"Expected empty list, got {path}"
    
    print("  ✓ RRT* algorithm tests passed!")

def test_prm_algorithm():
    """Test PRM algorithm."""
    print("Testing PRM algorithm...")
    
    grid = GridMap(20, 20)
    # Add some obstacles
    for i in range(5, 15):
        grid.add_obstacle(i, 10)
    
    start = (1, 1)
    goal = (18, 18)
    
    path = prm(start, goal, grid, num_samples=200, connection_radius=3.0)
    
    if path:  # Path found
        assert path[0] == start, f"Expected start {start}, got {path[0]}"
        assert path[-1] == goal, f"Expected goal {goal}, got {path[-1]}"
        
        # Path should be valid (no obstacles)
        for x, y in path:
            assert grid.is_free(x, y), f"Path goes through obstacle at ({x}, {y})"
    else:
        # If no path found, that's also acceptable
        assert path == [], f"Expected empty list, got {path}"
    
    print("  ✓ PRM algorithm tests passed!")

def test_simple_scenario():
    """Test algorithms with a simple, solvable scenario."""
    print("Testing algorithms with simple scenario...")
    
    # Create a simple grid with clear path
    simple_grid = GridMap(10, 10)
    simple_start = (1, 1)
    simple_goal = (8, 8)
    
    # Test RRT
    path = rrt(simple_start, simple_goal, simple_grid, 500)
    assert len(path) > 0, "RRT should find a path in simple scenario"
    if path:
        assert path[0] == simple_start, f"Expected {simple_start}, got {path[0]}"
        assert path[-1] == simple_goal, f"Expected {simple_goal}, got {path[-1]}"
    
    # Test RRT*
    path = rrt_star(simple_start, simple_goal, simple_grid, 500)
    assert len(path) > 0, "RRT* should find a path in simple scenario"
    if path:
        assert path[0] == simple_start, f"Expected {simple_start}, got {path[0]}"
        assert path[-1] == simple_goal, f"Expected {simple_goal}, got {path[-1]}"
    
    # Test PRM
    path = prm(simple_start, simple_goal, simple_grid, 100)
    assert len(path) > 0, "PRM should find a path in simple scenario"
    if path:
        assert path[0] == simple_start, f"Expected {simple_start}, got {path[0]}"
        assert path[-1] == simple_goal, f"Expected {simple_goal}, got {path[-1]}"
    
    print("  ✓ Simple scenario tests passed!")

def main():
    """Run all tests."""
    print("Running Advanced Algorithms Tests...")
    print("=" * 50)
    
    try:
        test_node_class()
        test_rrt_algorithm()
        test_rrt_star_algorithm()
        test_prm_algorithm()
        test_simple_scenario()
        
        print("=" * 50)
        print("All advanced algorithm tests passed! ✓")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
