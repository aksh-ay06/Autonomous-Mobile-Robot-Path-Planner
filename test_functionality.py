#!/usr/bin/env python3
"""
Simple test script to verify AMR Path Planner functionality.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from amr_path_planner import GridMap, PathPlanner, RobotAgent, DynamicObstacleMgr, Simulator


def test_basic_functionality():
    """Test basic functionality without visualization."""
    print("Testing AMR Path Planner basic functionality...")
    
    # Test GridMap
    print("1. Testing GridMap...")
    grid = GridMap(10, 10, {(3, 3), (4, 4), (5, 5)})
    assert grid.is_free(0, 0) == True
    assert grid.is_free(3, 3) == False
    assert len(grid.neighbors(1, 1)) == 4
    print("   ✓ GridMap working correctly")
    
    # Test PathPlanner
    print("2. Testing PathPlanner...")
    planner = PathPlanner('astar', grid=grid)
    path = planner.compute_path((0, 0), (9, 9))
    assert len(path) > 0
    assert path[0] == (0, 0)
    assert path[-1] == (9, 9)
    print(f"   ✓ PathPlanner found path of length {len(path)}")
    
    # Test RobotAgent
    print("3. Testing RobotAgent...")
    robot = RobotAgent((0, 0), planner)
    robot.plan_to((9, 9))
    assert robot.has_path()
    assert robot.goal == (9, 9)
    print("   ✓ RobotAgent working correctly")
    
    # Test DynamicObstacleMgr
    print("4. Testing DynamicObstacleMgr...")
    obstacle_mgr = DynamicObstacleMgr(grid)
    obstacle_mgr.add_obstacle(7, 7)
    assert obstacle_mgr.is_collision(7, 7)
    obstacle_mgr.update()  # Test movement
    print("   ✓ DynamicObstacleMgr working correctly")
    
    # Test basic simulation step
    print("5. Testing Simulator...")
    simulator = Simulator(grid, robot, obstacle_mgr, max_steps=10)
    stats_before = simulator.get_statistics()
    simulator.step_once()
    stats_after = simulator.get_statistics()
    assert stats_after['current_step'] == stats_before['current_step'] + 1
    print("   ✓ Simulator working correctly")
    
    print("\n✅ All tests passed! AMR Path Planner is working correctly.")
    return True


def test_pathfinding_algorithms():
    """Test both pathfinding algorithms."""
    print("\nTesting pathfinding algorithms...")
    
    grid = GridMap(8, 8, {(2, 2), (3, 2), (4, 2)})  # Simple barrier
    
    # Test Dijkstra
    dijkstra_planner = PathPlanner('dijkstra', grid=grid)
    dijkstra_path = dijkstra_planner.compute_path((0, 0), (7, 7))
    
    # Test A*
    astar_planner = PathPlanner('astar', grid=grid)
    astar_path = astar_planner.compute_path((0, 0), (7, 7))
    
    print(f"Dijkstra path length: {len(dijkstra_path)}")
    print(f"A* path length: {len(astar_path)}")
    
    # Both should find paths of the same length (optimal)
    assert len(dijkstra_path) == len(astar_path)
    assert len(dijkstra_path) > 0
    
    print("✅ Both algorithms found optimal paths of equal length!")
    return True


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_pathfinding_algorithms()
        print("\n🎉 All functionality tests completed successfully!")
        print("\nTo see the visual simulation, run: python examples/demo.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
