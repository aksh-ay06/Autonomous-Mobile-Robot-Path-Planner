"""
Comprehensive demo showcasing advanced features of the AMR Path Planner.

This script demonstrates:
1) Advanced algorithms (RRT, RRT*, PRM)
2) Path smoothing techniques (via smooth_path + direct methods)
3) Enhanced movement models (EnhancedGridMap + terrain costs)
4) Multi-robot coordination with space-time A* (reservation-table planning)
5) Comparison between different approaches
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path (so `python examples/advanced_demo.py` works)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_path_planner import PathPlanner, GridMap

# Advanced planners
from amr_path_planner.advanced_algorithms import rrt, rrt_star, prm

# Smoothing
from amr_path_planner.path_smoothing import (
    smooth_path,
    shortcut_smoothing,
    adaptive_smoothing,
    douglas_peucker_smoothing,
    bezier_smoothing,
    path_curvature_analysis,
)

# Enhanced grid
from amr_path_planner.enhanced_grid import EnhancedGridMap

# Multi-robot
from amr_path_planner.robot_agent import RobotAgent
from amr_path_planner.multi_robot_coordinator import MultiRobotCoordinator


Point = Tuple[int, int]


# ----------------------------
# Environment builders
# ----------------------------

def create_complex_environment() -> GridMap:
    """Create a complex environment for demonstration."""
    grid = GridMap(30, 25)

    # Wall with gaps
    for i in range(5, 25):
        if i not in [10, 15, 20]:  # Leave gaps
            grid.add_obstacle(i, 8)

    # L-shaped obstacle
    for i in range(8, 15):
        grid.add_obstacle(5, i)
    for i in range(5, 12):
        grid.add_obstacle(i, 14)

    # Scattered obstacles
    obstacles = [
        (18, 3), (19, 3), (18, 4), (19, 4),
        (25, 10), (26, 10), (25, 11), (26, 11),
        (12, 20), (13, 20), (14, 20),
        (20, 18), (21, 18), (22, 18), (20, 19), (21, 19), (22, 19)
    ]

    for x, y in obstacles:
        if 0 <= x < grid.width and 0 <= y < grid.height:
            grid.add_obstacle(x, y)

    return grid


def create_enhanced_terrain_environment() -> EnhancedGridMap:
    """Create an EnhancedGridMap with terrain costs + obstacles."""
    grid = EnhancedGridMap(20, 20)
    grid.movement_model = "8-connected"

    obstacles = [
        # Vertical wall with gap
        (8, 5), (8, 6), (8, 7), (8, 8), (8, 10), (8, 11), (8, 12), (8, 13),
        # Horizontal wall with gap
        (5, 15), (6, 15), (7, 15), (9, 15), (10, 15), (11, 15), (12, 15),
        # Scattered obstacles
        (15, 8), (15, 9), (16, 8), (16, 9)
    ]
    for x, y in obstacles:
        if 0 <= x < grid.width and 0 <= y < grid.height:
            grid.add_obstacle(x, y)

    # Terrain variation
    for i in range(grid.width):
        for j in range(grid.height):
            if grid.is_free(i, j):
                if 12 <= i <= 17 and 2 <= j <= 7:
                    grid.set_cell_cost(i, j, 2.5)  # rocky
                elif 2 <= i <= 6 and 10 <= j <= 15:
                    grid.set_cell_cost(i, j, 0.3)  # highway

    return grid


# ----------------------------
# Demos
# ----------------------------

def demo_advanced_algorithms() -> Dict[str, dict]:
    """Demonstrate advanced path planning algorithms."""
    print("=== Advanced Algorithms Demo ===")

    grid = create_complex_environment()
    start: Point = (2, 2)
    goal: Point = (27, 22)

    algorithms = {
        "RRT": lambda s, g: rrt(s, g, grid, max_iterations=2000, step_size=1.5, goal_bias=0.1),
        "RRT*": lambda s, g: rrt_star(s, g, grid, max_iterations=2000, step_size=1.5, goal_bias=0.1, search_radius=3.0),
        "PRM": lambda s, g: prm(s, g, grid, num_samples=300, connection_radius=4.0),
    }

    results: Dict[str, dict] = {}

    for name, planner_func in algorithms.items():
        print(f"\nTesting {name}...")
        start_time = time.time()
        path = planner_func(start, goal)
        planning_time = time.time() - start_time

        if path:
            path_length = sum(
                np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                for i in range(len(path) - 1)
            )
            print(f"  ✓ Path found! Length: {path_length:.2f}, Time: {planning_time:.3f}s")
            results[name] = {"path": path, "length": path_length, "time": planning_time, "success": True}
        else:
            print(f"  ✗ No path found. Time: {planning_time:.3f}s")
            results[name] = {"success": False, "time": planning_time}

    if any(r.get("success") for r in results.values()):
        visualize_algorithm_comparison(grid, start, goal, results)

    return results


def demo_path_smoothing() -> Dict[str, List[Point]]:
    """Demonstrate path smoothing techniques using updated smoothing APIs."""
    print("\n=== Path Smoothing Demo ===")

    grid = create_complex_environment()
    start: Point = (2, 2)
    goal: Point = (27, 22)

    planner = PathPlanner(algorithm="astar", grid=grid)
    original_path = planner.compute_path(start, goal)

    if not original_path:
        print("Could not find initial path for smoothing demo")
        return {}

    print(f"Original path length: {len(original_path)} points")
    original_analysis = path_curvature_analysis(original_path)
    print(f"Original path analysis: {original_analysis}")

    # Updated smoothing methods (matching your new implementations)
    smoothing_methods = {
        # Direct methods
        "Shortcut (direct)": lambda p: shortcut_smoothing(p, grid, max_iterations=150),
        "Adaptive (direct)": lambda p: adaptive_smoothing(p, grid, curvature_threshold=0.9, lookahead=12),
        "Douglas-Peucker (direct)": lambda p: douglas_peucker_smoothing(p, epsilon=0.8),

        # Via unified smooth_path() wrapper
        "Shortcut (smooth_path)": lambda p: smooth_path(p, grid, method="shortcut", max_iterations=150),
        "Adaptive (smooth_path)": lambda p: smooth_path(p, grid, method="adaptive", curvature_threshold=0.9, lookahead=12),
        "Douglas-Peucker (smooth_path)": lambda p: smooth_path(p, grid, method="douglas_peucker", epsilon=0.8),
        "Bezier-like (smooth_path)": lambda p: smooth_path(p, grid, method="bezier", iterations=2),
        "Spline-like (smooth_path)": lambda p: smooth_path(p, grid, method="spline", smoothing_factor=0.6, upsample=6),
    }

    smoothed_paths: Dict[str, List[Point]] = {}

    for method_name, method_func in smoothing_methods.items():
        try:
            print(f"\nApplying {method_name}...")
            sp = method_func(original_path)
            analysis = path_curvature_analysis(sp)
            print(f"  Points: {len(sp)}, Analysis: {analysis}")
            smoothed_paths[method_name] = sp
        except Exception as e:
            print(f"  Error in {method_name}: {e}")

    if smoothed_paths:
        visualize_smoothing_comparison(grid, original_path, smoothed_paths)

    return smoothed_paths


def demo_enhanced_movement() -> Dict[str, dict]:
    """Demonstrate EnhancedGridMap movement models + terrain costs."""
    print("\n=== Enhanced Movement Models Demo ===")

    grid = EnhancedGridMap(15, 15)

    # Obstacles block
    obstacles = [(6, 6), (6, 7), (6, 8), (7, 6), (7, 7), (7, 8), (8, 6), (8, 7), (8, 8)]
    for x, y in obstacles:
        grid.add_obstacle(x, y)

    # Terrain costs
    for i in range(15):
        for j in range(15):
            if grid.is_free(i, j):
                if 2 <= i <= 4:
                    grid.set_cell_cost(i, j, 3.0)   # difficult
                elif 10 <= i <= 12:
                    grid.set_cell_cost(i, j, 0.5)   # easy

    start: Point = (1, 1)
    goal: Point = (13, 13)

    movement_models = ["4-connected", "8-connected", "knight"]
    results: Dict[str, dict] = {}

    for movement_model in movement_models:
        print(f"\nTesting {movement_model} movement...")
        grid.movement_model = movement_model

        planner = PathPlanner(algorithm="astar", grid=grid)
        path = planner.compute_path(start, goal)

        if path:
            total_cost = sum(grid.get_movement_cost(path[i], path[i + 1]) for i in range(len(path) - 1))
            print(f"  ✓ Path found! Points: {len(path)}, Total cost: {total_cost:.2f}")
            results[movement_model] = {"path": path, "cost": total_cost, "success": True}
        else:
            print("  ✗ No path found")
            results[movement_model] = {"success": False}

    if any(r.get("success") for r in results.values()):
        visualize_movement_comparison(grid, start, goal, results)

    return results


def demo_integrated_features() -> None:
    """Demonstrate enhanced grid + A* + smoothing using the updated planner API."""
    print("\n=== Integrated Features Demo ===")

    grid = create_enhanced_terrain_environment()
    start: Point = (2, 2)
    goal: Point = (17, 17)

    print("Planning with PathPlanner (A* + Enhanced Grid + terrain costs)...")

    planner = PathPlanner(algorithm="astar", grid=grid)
    path = planner.compute_path(start, goal)

    if not path:
        print("No path found")
        return

    print(f"Initial path: {len(path)} points")

    print("Applying smoothing via smooth_path (collision-safe)...")
    smoothed = smooth_path(path, grid, method="shortcut", max_iterations=200)

    original_analysis = path_curvature_analysis(path)
    smoothed_analysis = path_curvature_analysis(smoothed)

    print(f"Original analysis: {original_analysis}")
    print(f"Smoothed analysis: {smoothed_analysis}")

    visualize_integrated_demo(grid, start, goal, path, smoothed)


def demo_multi_robot_coordination() -> None:
    """Demonstrate multi-robot space-time A* coordination (reservation-table planning)."""
    print("\n=== Multi-Robot Coordination Demo (Space-Time A*) ===")

    grid = create_complex_environment()

    # Each robot uses a planner for fallback/local planning if desired,
    # but coordinator does the core space-time planning.
    planner = PathPlanner(algorithm="astar", grid=grid)

    robots = [
        RobotAgent(position=(2, 2), planner=planner),
        RobotAgent(position=(2, 22), planner=planner),
        RobotAgent(position=(27, 2), planner=planner),
    ]
    goals = [(27, 22), (27, 2), (2, 22)]

    coordinator = MultiRobotCoordinator(grid=grid, coordination_mode="cooperative", horizon=18)

    for i, r in enumerate(robots):
        rid = coordinator.add_robot(r, priority=(10 - i))  # simple priorities
        coordinator.set_robot_goal(rid, goals[i])

    coordinator.plan_coordinated_paths()

    # Step simulation
    max_steps = 40
    trajectories: Dict[int, List[Point]] = {i: [robots[i].position] for i in range(len(robots))}

    for _ in range(max_steps):
        coordinator.step_all_robots()
        for i, r in enumerate(robots):
            trajectories[i].append(r.position)

        if all(r.is_at_goal() for r in robots):
            break

    stats = coordinator.get_coordination_statistics()
    print(f"Coordination stats: {stats}")

    visualize_multi_robot_trajectories(grid, robots, goals, trajectories)


# ----------------------------
# Visualization helpers
# ----------------------------

def visualize_algorithm_comparison(grid: GridMap, start: Point, goal: Point, results: Dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Advanced Algorithms Comparison", fontsize=16)

    algorithm_names = ["RRT", "RRT*", "PRM"]
    colors = ["red", "blue", "green"]

    for idx, (name, color) in enumerate(zip(algorithm_names, colors)):
        ax = axes[idx]

        # obstacles
        for i in range(grid.width):
            for j in range(grid.height):
                if not grid.is_free(i, j):
                    ax.plot(i, j, "ks", markersize=3)

        if name in results and results[name].get("success"):
            path = results[name]["path"]
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            ax.plot(x_coords, y_coords, color=color, linewidth=2, label=f"{name} Path")

        ax.plot(start[0], start[1], "go", markersize=8, label="Start")
        ax.plot(goal[0], goal[1], "ro", markersize=8, label="Goal")

        ax.set_title(name)
        ax.set_xlim(-1, grid.width)
        ax.set_ylim(-1, grid.height)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("advanced_algorithms_demo.png", dpi=150, bbox_inches="tight")
    plt.show()


def visualize_smoothing_comparison(grid: GridMap, original_path: List[Point], smoothed_paths: Dict[str, List[Point]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Path Smoothing Techniques Comparison", fontsize=16)

    ax = axes[0, 0]
    draw_grid_and_path(ax, grid, original_path, "Original Path", "black")

    methods = list(smoothed_paths.keys())
    colors = ["red", "blue", "green", "orange", "purple", "brown"]
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for idx, method in enumerate(methods[: len(positions)]):
        row, col = positions[idx]
        ax = axes[row, col]
        color = colors[idx % len(colors)]
        draw_grid_and_path(ax, grid, smoothed_paths[method], method, color)

    # Hide unused axes
    used = 1 + min(len(methods), len(positions))
    all_slots = [(0, 0)] + positions
    for slot_idx in range(used, len(all_slots)):
        r, c = all_slots[slot_idx]
        axes[r, c].set_visible(False)

    plt.tight_layout()
    plt.savefig("path_smoothing_demo.png", dpi=150, bbox_inches="tight")
    plt.show()


def visualize_movement_comparison(grid: EnhancedGridMap, start: Point, goal: Point, results: Dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Movement Models Comparison", fontsize=16)

    movement_names = ["4-connected", "8-connected", "knight"]
    colors = ["red", "blue", "green"]

    for idx, (name, color) in enumerate(zip(movement_names, colors)):
        ax = axes[idx]

        for i in range(grid.width):
            for j in range(grid.height):
                if not grid.is_free(i, j):
                    ax.plot(i, j, "ks", markersize=4)
                else:
                    cost = grid.get_cell_cost(i, j)
                    if cost > 1.5:
                        ax.plot(i, j, "rs", markersize=2, alpha=0.5)
                    elif cost < 0.8:
                        ax.plot(i, j, "gs", markersize=2, alpha=0.5)

        if name in results and results[name].get("success"):
            path = results[name]["path"]
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            ax.plot(x_coords, y_coords, color=color, linewidth=2, label=f"{name} Path")

        ax.plot(start[0], start[1], "go", markersize=8, label="Start")
        ax.plot(goal[0], goal[1], "ro", markersize=8, label="Goal")

        ax.set_title(f"{name} Movement")
        ax.set_xlim(-1, grid.width)
        ax.set_ylim(-1, grid.height)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("movement_models_demo.png", dpi=150, bbox_inches="tight")
    plt.show()


def visualize_integrated_demo(grid: EnhancedGridMap, start: Point, goal: Point, original_path: List[Point], smoothed_path: List[Point]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Integrated Features: Enhanced Grid + A* + Path Smoothing", fontsize=14)

    draw_enhanced_grid_and_path(ax1, grid, original_path, start, goal, "Original A* Path", "blue")
    ax1.set_title("Original Path")

    draw_enhanced_grid_and_path(ax2, grid, smoothed_path, start, goal, "Smoothed Path", "green")
    ax2.set_title("Smoothed Path")

    plt.tight_layout()
    plt.savefig("integrated_features_demo.png", dpi=150, bbox_inches="tight")
    plt.show()


def visualize_multi_robot_trajectories(
    grid: GridMap,
    robots: List[RobotAgent],
    goals: List[Point],
    trajectories: Dict[int, List[Point]],
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.set_title("Multi-Robot Space-Time A* (trajectories)", fontsize=14)

    # obstacles
    for i in range(grid.width):
        for j in range(grid.height):
            if not grid.is_free(i, j):
                ax.plot(i, j, "ks", markersize=3)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for rid, traj in trajectories.items():
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax.plot(xs, ys, linewidth=2, label=f"Robot {rid}", color=colors[rid % len(colors)])

        # Start + Goal markers
        ax.plot(xs[0], ys[0], "o", markersize=7, color=colors[rid % len(colors)])
        ax.plot(goals[rid][0], goals[rid][1], "X", markersize=9, color=colors[rid % len(colors)])

    ax.set_xlim(-1, grid.width)
    ax.set_ylim(-1, grid.height)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("multi_robot_coordination_demo.png", dpi=150, bbox_inches="tight")
    plt.show()


def draw_grid_and_path(ax, grid: GridMap, path: List[Point], title: str, color: str) -> None:
    for i in range(grid.width):
        for j in range(grid.height):
            if not grid.is_free(i, j):
                ax.plot(i, j, "ks", markersize=3)

    if path:
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        ax.plot(x_coords, y_coords, color=color, linewidth=2, label=title)
        ax.plot(path[0][0], path[0][1], "go", markersize=6, label="Start")
        ax.plot(path[-1][0], path[-1][1], "ro", markersize=6, label="Goal")

    ax.set_xlim(-1, grid.width)
    ax.set_ylim(-1, grid.height)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")


def draw_enhanced_grid_and_path(
    ax,
    grid: EnhancedGridMap,
    path: List[Point],
    start: Point,
    goal: Point,
    title: str,
    color: str,
) -> None:
    for i in range(grid.width):
        for j in range(grid.height):
            if not grid.is_free(i, j):
                ax.plot(i, j, "ks", markersize=4)
            else:
                cost = grid.get_cell_cost(i, j)
                if cost > 1.5:
                    ax.plot(i, j, "rs", markersize=2, alpha=0.6)
                elif cost < 0.8:
                    ax.plot(i, j, "gs", markersize=2, alpha=0.6)

    if path:
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        ax.plot(x_coords, y_coords, color=color, linewidth=2, label=title)

    ax.plot(start[0], start[1], "go", markersize=8, label="Start")
    ax.plot(goal[0], goal[1], "ro", markersize=8, label="Goal")

    ax.set_xlim(-1, grid.width)
    ax.set_ylim(-1, grid.height)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    print("AMR Path Planner - Advanced Features Demo")
    print("=" * 55)

    try:
        demo_advanced_algorithms()
        demo_path_smoothing()
        demo_enhanced_movement()
        demo_integrated_features()
        demo_multi_robot_coordination()

        print("\n" + "=" * 55)
        print("All demos completed successfully!")
        print("Generated PNG files:")
        print(" - advanced_algorithms_demo.png")
        print(" - path_smoothing_demo.png")
        print(" - movement_models_demo.png")
        print(" - integrated_features_demo.png")
        print(" - multi_robot_coordination_demo.png")

    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Install: pip install matplotlib numpy (and optionally scipy for spline smoothing)")
    except Exception as e:
        print(f"Error running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
