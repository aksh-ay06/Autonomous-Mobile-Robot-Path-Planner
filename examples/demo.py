"""
Demo script for AMR Path Planner.
Demonstrates the autonomous mobile robot path planning system.

Updates in this version:
- Uses the updated GridMap strict obstacle policy safely (obstacles provided are in-bounds).
- Ensures dynamic obstacles are added BEFORE sampling a random start cell.
- Uses the updated RobotAgent API (planner passed by name is supported, but we use keyword style).
- Keeps behavior identical: A* planning + dynamic obstacles + Simulator visualization (+ optional GIF).
"""

from __future__ import annotations

import os
import random
import sys
from typing import List, Tuple

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amr_path_planner import GridMap, PathPlanner, DynamicObstacleMgr, RobotAgent, Simulator

Point = Tuple[int, int]


def create_sample_environment() -> GridMap:
    """Create a sample environment with obstacles."""
    width, height = 20, 15
    static_obstacles: set[Point] = set()

    # Vertical wall
    for y in range(3, 12):
        static_obstacles.add((8, y))

    # Horizontal barriers
    for x in range(2, 7):
        static_obstacles.add((x, 5))
    for x in range(12, 18):
        static_obstacles.add((x, 8))

    # Scattered obstacles
    scattered = [(3, 2), (15, 3), (6, 11), (17, 13), (2, 13)]
    static_obstacles.update(scattered)

    # GridMap is strict now: all obstacles must be in bounds.
    return GridMap(width, height, static_obstacles)


def _pick_random_free_cell(grid: GridMap, blocked: set[Point], forbidden: set[Point]) -> Point:
    """Pick a random free cell that is not blocked and not in forbidden set."""
    all_cells = [(x, y) for x in range(grid.width) for y in range(grid.height)]
    candidates = [c for c in all_cells if c not in blocked and c not in forbidden and grid.is_free(*c)]
    if not candidates:
        raise RuntimeError("No free cells available for robot start position!")
    return random.choice(candidates)


def main() -> None:
    """Main demo function."""
    print("AMR Path Planner Demo")
    print("====================")

    grid = create_sample_environment()
    print(f"Created {grid.width}x{grid.height} grid with {len(grid.static_obstacles)} static obstacles")

    # Create path planner (A* by default)
    planner = PathPlanner(algorithm="astar", grid=grid)
    print("Created A* path planner")

    # Dynamic obstacles
    obstacle_mgr = DynamicObstacleMgr(grid, movement_probability=0.6)
    dynamic_positions: List[Point] = [(5, 3), (10, 6), (14, 10), (7, 12)]
    for x, y in dynamic_positions:
        if grid.is_free(x, y):  # only place on free cells
            obstacle_mgr.add_obstacle(x, y)
    print(f"Added {len(dynamic_positions)} dynamic obstacles (requested)")

    # Define goal
    goal_pos: Point = (18, 13)
    if not grid.is_free(*goal_pos):
        raise RuntimeError(f"Goal {goal_pos} is not in a free cell. Please change the goal location.")

    # Choose a random free start position AFTER dynamic obstacles exist
    occupied: set[Point] = set(grid.static_obstacles)
    occupied.update(obstacle_mgr.get_obstacle_positions())

    start_pos = _pick_random_free_cell(grid, blocked=occupied, forbidden={goal_pos})
    robot = RobotAgent(position=start_pos, planner=planner)
    print(f"Created robot agent at random position {start_pos}")

    # Plan to goal
    robot.plan_to(goal_pos)
    print(f"Robot planning path to goal {goal_pos}")

    if robot.has_path():
        print(f"Initial path length: {len(robot.path)} points")
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
        simulator.run(visualize=True, save_gif=False)

        stats = simulator.get_statistics()
        print("\nSimulation completed!")
        print(f"Steps taken: {stats.get('current_step')}")
        print(f"Goal reached: {stats.get('goal_reached')}")
        print(f"Final robot position: {stats.get('robot_position')}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError during simulation: {e}")


def demo_with_gif() -> None:
    """Demo that saves animation as GIF."""
    print("AMR Path Planner Demo - GIF Export")
    print("==================================")

    grid = GridMap(12, 8, {(4, 3), (4, 4), (5, 4), (7, 2), (8, 6)})
    planner = PathPlanner(algorithm="astar", grid=grid)
    robot = RobotAgent(position=(1, 1), planner=planner)

    obstacle_mgr = DynamicObstacleMgr(grid, movement_probability=0.5)
    # Only add dynamic obstacles if the cells are free
    if grid.is_free(3, 2):
        obstacle_mgr.add_obstacle(3, 2)
    if grid.is_free(9, 5):
        obstacle_mgr.add_obstacle(9, 5)

    goal: Point = (10, 6)
    if not grid.is_free(*goal):
        raise RuntimeError(f"Goal {goal} is blocked by a static obstacle.")

    robot.plan_to(goal)

    simulator = Simulator(grid, agent=robot, obstacle_mgr=obstacle_mgr, step_delay=0.2, max_steps=500)

    print("Generating GIF animation...")
    simulator.run(visualize=True, save_gif=True, gif_filename="amr_demo.gif")
    print("GIF saved as 'amr_demo.gif'")


if __name__ == "__main__":
    main()

    # Uncomment to also generate a GIF
    demo_with_gif()
