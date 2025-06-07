"""
Demo script for AMR Path Planner.
Demonstrates the autonomous mobile robot path planning system.
"""

import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from amr_path_planner import GridMap, PathPlanner, DynamicObstacleMgr, RobotAgent, Simulator


def create_sample_environment():
    """Create a sample environment with obstacles."""
    # Create 20x15 grid
    width, height = 20, 15
    
    # Add some static obstacles (walls and barriers)
    static_obstacles = set()
    
    # Vertical wall
    for y in range(3, 12):
        static_obstacles.add((8, y))
    
    # Horizontal barriers
    for x in range(2, 7):
        static_obstacles.add((x, 5))
    
    for x in range(12, 18):
        static_obstacles.add((x, 8))
    
    # Some scattered obstacles
    scattered = [(3, 2), (15, 3), (6, 11), (17, 13), (2, 13)]
    static_obstacles.update(scattered)
    
    return GridMap(width, height, static_obstacles)


def main():
    """Main demo function."""
    print("AMR Path Planner Demo")
    print("====================")
    
    # Create environment
    grid = create_sample_environment()
    print(f"Created {grid.width}x{grid.height} grid with {len(grid.static_obstacles)} static obstacles")
    
    # Create path planner (using A* by default)
    planner = PathPlanner('astar', grid=grid)
    print("Created A* path planner")
    
    # Create robot agent
    start_pos = (1, 1)
    robot = RobotAgent(start_pos, planner)
    print(f"Created robot agent at position {start_pos}")
    
    # Create dynamic obstacle manager
    obstacle_mgr = DynamicObstacleMgr(grid, movement_probability=0.6)
    
    # Add some dynamic obstacles
    dynamic_positions = [(5, 3), (10, 6), (14, 10), (7, 12)]
    for pos in dynamic_positions:
        obstacle_mgr.add_obstacle(pos[0], pos[1])
    print(f"Added {len(dynamic_positions)} dynamic obstacles")
    
    # Set goal for robot
    goal_pos = (18, 13)
    robot.plan_to(goal_pos)
    print(f"Robot planning path to goal {goal_pos}")
    
    if robot.has_path():
        print(f"Initial path length: {len(robot.path)} steps")
    else:
        print("No initial path found!")
        return
    
    # Create simulator
    simulator = Simulator(grid, agent=robot, obstacle_mgr=obstacle_mgr, step_delay=0.2, max_steps=500)
    print("Created simulator")
    
    print("\nStarting simulation...")
    print("Close the plot window to end the simulation.")
    print("The robot (blue) will navigate to the goal (green)")
    print("while avoiding static obstacles (black) and dynamic obstacles (red)")
    
    try:
        # Run simulation with visualization
        simulator.run(visualize=True, save_gif=False)
        
        # Print final statistics
        stats = simulator.get_statistics()
        print(f"\nSimulation completed!")
        print(f"Steps taken: {stats['current_step']}")
        print(f"Goal reached: {stats['goal_reached']}")
        print(f"Final robot position: {stats['robot_position']}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError during simulation: {e}")


def demo_with_gif():
    """Demo that saves animation as GIF."""
    print("AMR Path Planner Demo - GIF Export")
    print("==================================")
    
    # Create smaller environment for GIF
    grid = GridMap(12, 8, {(4, 3), (4, 4), (5, 4), (7, 2), (8, 6)})
    planner = PathPlanner('astar', grid=grid)
    robot = RobotAgent((1, 1), planner)
    
    # Add fewer dynamic obstacles for cleaner visualization
    obstacle_mgr = DynamicObstacleMgr(grid, movement_probability=0.5)
    obstacle_mgr.add_obstacle(3, 2)
    obstacle_mgr.add_obstacle(9, 5)
    
    robot.plan_to((10, 6))
    
    simulator = Simulator(grid, robot, obstacle_mgr, step_delay=0.3, max_steps=100)
    
    print("Generating GIF animation...")
    simulator.run(visualize=True, save_gif=True, gif_filename="amr_demo.gif")
    print("GIF saved as 'amr_demo.gif'")


if __name__ == "__main__":
    # Run main demo
    main()
    
    # Uncomment the line below to also generate a GIF
    # demo_with_gif()
