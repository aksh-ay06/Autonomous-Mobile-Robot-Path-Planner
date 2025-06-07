"""
Comprehensive demo showcasing all the new advanced features of the AMR Path Planner.

This script demonstrates:
1. Advanced algorithms (RRT, RRT*, PRM)
2. Path smoothing techniques
3. Enhanced movement models
4. Comparison between different approaches
"""

import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_path_planner import PathPlanner, GridMap
from amr_path_planner.advanced_algorithms import rrt, rrt_star, prm
from amr_path_planner.path_smoothing import (
    shortcut_smoothing, bezier_smoothing, adaptive_smoothing,
    douglas_peucker_smoothing, path_curvature_analysis
)
from amr_path_planner.enhanced_grid import EnhancedGridMap


def create_complex_environment():
    """Create a complex environment for demonstration."""
    grid = GridMap(30, 25)
    
    # Add various obstacle patterns
    
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


def demo_advanced_algorithms():
    """Demonstrate advanced path planning algorithms."""
    print("=== Advanced Algorithms Demo ===")
    
    grid = create_complex_environment()
    start = (2, 2)
    goal = (27, 22)
    
    # Use function-based planners
    algorithms = {
        'RRT': lambda s, g: rrt(s, g, grid, max_iterations=2000, step_size=1.5, goal_bias=0.1),
        'RRT*': lambda s, g: rrt_star(s, g, grid, max_iterations=2000, step_size=1.5, goal_bias=0.1, search_radius=3.0),
        'PRM': lambda s, g: prm(s, g, grid, num_samples=300, connection_radius=4.0)
    }
    
    # Plan paths and measure performance
    results = {}
    
    for name, planner_func in algorithms.items():
        print(f"\nTesting {name}...")
        start_time = time.time()
        path = planner_func(start, goal)
        planning_time = time.time() - start_time
        
        if path:
            path_length = sum(
                np.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
                for i in range(len(path) - 1)
            )
            print(f"  ✓ Path found! Length: {path_length:.2f}, Time: {planning_time:.3f}s")
            results[name] = {
                'path': path,
                'length': path_length,
                'time': planning_time,
                'success': True
            }
        else:
            print(f"  ✗ No path found. Time: {planning_time:.3f}s")
            results[name] = {'success': False, 'time': planning_time}
    
    # Visualize results
    if any(result['success'] for result in results.values()):
        visualize_algorithm_comparison(grid, start, goal, results)
    
    return results


def demo_path_smoothing():
    """Demonstrate path smoothing techniques."""
    print("\n=== Path Smoothing Demo ===")
    
    grid = create_complex_environment()
    start = (2, 2)
    goal = (27, 22)
    
    # Get a path using A* first
    planner = PathPlanner(algorithm='astar', grid=grid)
    original_path = planner.compute_path(start, goal)
    
    if not original_path:
        print("Could not find initial path for smoothing demo")
        return
    
    print(f"Original path length: {len(original_path)} points")
    original_analysis = path_curvature_analysis(original_path)
    print(f"Original path analysis: {original_analysis}")
    
    # Apply different smoothing techniques
    smoothing_methods = {
        'Shortcut': lambda p: shortcut_smoothing(p, grid, max_iterations=100),
        'Bezier': lambda p: bezier_smoothing(p, num_points=20),
        'Adaptive': lambda p: adaptive_smoothing(p, curvature_threshold=1.0, grid=grid),
        'Douglas-Peucker': lambda p: douglas_peucker_smoothing(p, epsilon=0.8)
    }
    
    smoothed_paths = {}
    
    for method_name, method_func in smoothing_methods.items():
        try:
            print(f"\nApplying {method_name} smoothing...")
            smoothed_path = method_func(original_path)
            analysis = path_curvature_analysis(smoothed_path)
            
            print(f"  Points: {len(smoothed_path)}, Analysis: {analysis}")
            smoothed_paths[method_name] = smoothed_path
            
        except ImportError as e:
            print(f"  Skipping {method_name}: {e}")
        except Exception as e:
            print(f"  Error in {method_name}: {e}")
    
    # Visualize smoothing results
    if smoothed_paths:
        visualize_smoothing_comparison(grid, original_path, smoothed_paths)
    
    return smoothed_paths


def demo_enhanced_movement():
    """Demonstrate enhanced movement models."""
    print("\n=== Enhanced Movement Models Demo ===")
    
    # Create a simpler grid for movement demonstration
    grid = EnhancedGridMap(15, 15)
    
    # Add some obstacles
    obstacles = [(6, 6), (6, 7), (6, 8), (7, 6), (7, 7), (7, 8), (8, 6), (8, 7), (8, 8)]
    for x, y in obstacles:
        grid.add_obstacle(x, y)
    
    # Add terrain costs
    for i in range(15):
        for j in range(15):
            if grid.is_free(i, j):
                # Create varied terrain
                if 2 <= i <= 4:
                    grid.set_cell_cost(i, j, 3.0)  # Difficult terrain
                elif 10 <= i <= 12:
                    grid.set_cell_cost(i, j, 0.5)  # Easy terrain
    
    start = (1, 1)
    goal = (13, 13)
    
    movement_models = [
        '4-connected',
        '8-connected',
        'knight'
    ]
    
    results = {}
    
    for movement_model in movement_models:
        print(f"\nTesting {movement_model} movement...")
        grid.movement_model = movement_model
        
        # Use PathPlanner with enhanced grid
        planner = PathPlanner(algorithm='astar', grid=grid)
        path = planner.compute_path(start, goal)
        
        if path:
            # Calculate total movement cost
            total_cost = sum(
                grid.get_movement_cost(path[i], path[i+1])
                for i in range(len(path) - 1)
            )
            print(f"  ✓ Path found! Points: {len(path)}, Total cost: {total_cost:.2f}")
            results[movement_model] = {
                'path': path,
                'cost': total_cost,
                'success': True
            }
        else:
            print(f"  ✗ No path found")
            results[movement_model] = {'success': False}
    
    # Visualize movement comparison
    if any(result['success'] for result in results.values()):
        visualize_movement_comparison(grid, start, goal, results)
    
    return results


def demo_integrated_features():
    """Demonstrate integration of all features together."""
    print("\n=== Integrated Features Demo ===")
    
    # Create enhanced grid with terrain
    grid = EnhancedGridMap(20, 20)
    grid.movement_model = '8-connected'
    
    # Add complex obstacle pattern
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
    
    # Add terrain variation
    for i in range(20):
        for j in range(20):
            if grid.is_free(i, j):
                # Create terrain patterns
                if 12 <= i <= 17 and 2 <= j <= 7:
                    grid.set_cell_cost(i, j, 2.5)  # Rocky area
                elif 2 <= i <= 6 and 10 <= j <= 15:
                    grid.set_cell_cost(i, j, 0.3)  # Highway
    
    start = (2, 2)
    goal = (17, 17)
    
    print("Planning with advanced PathPlanner (A* + Enhanced Grid)...")
    
    # Use the enhanced PathPlanner
    planner = PathPlanner(algorithm='astar', grid=grid)
    path = planner.compute_path(start, goal)
    
    if path:
        print(f"Initial path: {len(path)} points")
        
        # Apply smoothing
        print("Applying smoothing...")
        from amr_path_planner.path_smoothing import smooth_path
        smoothed_path = smooth_path(path, grid)
        
        if smoothed_path:
            print(f"Smoothed path: {len(smoothed_path)} points")
            
            # Analyze both paths
            original_analysis = path_curvature_analysis(path)
            smoothed_analysis = path_curvature_analysis(smoothed_path)
            
            print(f"Original analysis: {original_analysis}")
            print(f"Smoothed analysis: {smoothed_analysis}")
            
            # Visualize integrated result
            visualize_integrated_demo(grid, start, goal, path, smoothed_path)
        else:
            print("Smoothing failed")
    else:
        print("No path found")


def visualize_algorithm_comparison(grid, start, goal, results):
    """Visualize comparison of different algorithms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Advanced Algorithms Comparison', fontsize=16)
    
    algorithm_names = ['RRT', 'RRT*', 'PRM']
    colors = ['red', 'blue', 'green']
    
    for idx, (name, color) in enumerate(zip(algorithm_names, colors)):
        ax = axes[idx]
        
        # Draw grid
        for i in range(grid.width):
            for j in range(grid.height):
                if not grid.is_free(i, j):
                    ax.plot(i, j, 'ks', markersize=3)
        
        # Draw path if found
        if name in results and results[name]['success']:
            path = results[name]['path']
            if path:
                x_coords = [p[0] for p in path]
                y_coords = [p[1] for p in path]
                ax.plot(x_coords, y_coords, color=color, linewidth=2, label=f'{name} Path')
                
                # Mark start and goal
                ax.plot(start[0], start[1], 'go', markersize=8, label='Start')
                ax.plot(goal[0], goal[1], 'ro', markersize=8, label='Goal')
        
        ax.set_title(f'{name}')
        ax.set_xlim(-1, grid.width)
        ax.set_ylim(-1, grid.height)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('advanced_algorithms_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_smoothing_comparison(grid, original_path, smoothed_paths):
    """Visualize comparison of different smoothing techniques."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Path Smoothing Techniques Comparison', fontsize=16)
    
    # Original path in first subplot
    ax = axes[0, 0]
    draw_grid_and_path(ax, grid, original_path, 'Original Path', 'black')
    
    # Smoothed paths
    methods = list(smoothed_paths.keys())
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    
    for idx, (method, path) in enumerate(smoothed_paths.items()):
        if idx < len(positions):
            row, col = positions[idx]
            ax = axes[row, col]
            color = colors[idx % len(colors)]
            draw_grid_and_path(ax, grid, path, f'{method} Smoothing', color)
    
    # Hide unused subplots
    for idx in range(len(smoothed_paths) + 1, 6):
        row, col = positions[idx - 1]
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('path_smoothing_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_movement_comparison(grid, start, goal, results):
    """Visualize comparison of different movement models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Movement Models Comparison', fontsize=16)
    
    movement_names = ['4-connected', '8-connected', 'knight']
    colors = ['red', 'blue', 'green']
    
    for idx, (name, color) in enumerate(zip(movement_names, colors)):
        ax = axes[idx]
        
        # Draw grid with terrain costs
        for i in range(grid.width):
            for j in range(grid.height):
                if not grid.is_free(i, j):
                    ax.plot(i, j, 'ks', markersize=4)
                else:
                    cost = grid.get_cell_cost(i, j)
                    if cost > 1.5:
                        ax.plot(i, j, 'rs', markersize=2, alpha=0.5)  # Difficult terrain
                    elif cost < 0.8:
                        ax.plot(i, j, 'gs', markersize=2, alpha=0.5)  # Easy terrain
        
        # Draw path if found
        if name in results and results[name]['success']:
            path = results[name]['path']
            if path:
                x_coords = [p[0] for p in path]
                y_coords = [p[1] for p in path]
                ax.plot(x_coords, y_coords, color=color, linewidth=2, label=f'{name} Path')
        
        # Mark start and goal
        ax.plot(start[0], start[1], 'go', markersize=8, label='Start')
        ax.plot(goal[0], goal[1], 'ro', markersize=8, label='Goal')
        
        ax.set_title(f'{name} Movement')
        ax.set_xlim(-1, grid.width)
        ax.set_ylim(-1, grid.height)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('movement_models_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_integrated_demo(grid, start, goal, original_path, smoothed_path):
    """Visualize the integrated demo with all features."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Integrated Features: Enhanced Grid + A* + Path Smoothing', fontsize=14)
    
    # Original path
    draw_enhanced_grid_and_path(ax1, grid, original_path, start, goal, 'Original A* Path', 'blue')
    ax1.set_title('Original Path')
    
    # Smoothed path
    draw_enhanced_grid_and_path(ax2, grid, smoothed_path, start, goal, 'Smoothed Path', 'green')
    ax2.set_title('Smoothed Path')
    
    plt.tight_layout()
    plt.savefig('integrated_features_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


def draw_grid_and_path(ax, grid, path, title, color):
    """Helper function to draw grid and path."""
    # Draw obstacles
    for i in range(grid.width):
        for j in range(grid.height):
            if not grid.is_free(i, j):
                ax.plot(i, j, 'ks', markersize=3)
    
    # Draw path
    if path:
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        ax.plot(x_coords, y_coords, color=color, linewidth=2, label=title)
        
        # Mark start and goal
        ax.plot(path[0][0], path[0][1], 'go', markersize=6, label='Start')
        ax.plot(path[-1][0], path[-1][1], 'ro', markersize=6, label='Goal')
    
    ax.set_xlim(-1, grid.width)
    ax.set_ylim(-1, grid.height)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')


def draw_enhanced_grid_and_path(ax, grid, path, start, goal, title, color):
    """Helper function to draw enhanced grid with terrain costs and path."""
    # Draw terrain and obstacles
    for i in range(grid.width):
        for j in range(grid.height):
            if not grid.is_free(i, j):
                ax.plot(i, j, 'ks', markersize=4)
            else:
                cost = grid.get_cell_cost(i, j)
                if cost > 1.5:
                    ax.plot(i, j, 'rs', markersize=2, alpha=0.6)  # Difficult terrain
                elif cost < 0.8:
                    ax.plot(i, j, 'gs', markersize=2, alpha=0.6)  # Easy terrain
    
    # Draw path
    if path:
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        ax.plot(x_coords, y_coords, color=color, linewidth=2, label=title)
    
    # Mark start and goal
    ax.plot(start[0], start[1], 'go', markersize=8, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=8, label='Goal')
    
    ax.set_xlim(-1, grid.width)
    ax.set_ylim(-1, grid.height)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')


def main():
    """Run all demos."""
    print("AMR Path Planner - Advanced Features Demo")
    print("=" * 50)
    
    try:
        # Run individual demos
        demo_advanced_algorithms()
        demo_path_smoothing()
        demo_enhanced_movement()
        demo_integrated_features()
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        print("Check the generated PNG files for visualizations.")
        
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Please install required packages: pip install matplotlib numpy scipy")
    except Exception as e:
        print(f"Error running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
