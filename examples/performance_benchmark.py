"""
Performance comparison demo between traditional and advanced algorithms.

This script benchmarks different algorithms and provides detailed performance analysis.
"""

import sys
import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_path_planner import PathPlanner, GridMap
##from amr_path_planner.advanced_algorithms import RRTPlanner, RRTStarPlanner, PRMPlanner
from amr_path_planner.advanced_algorithms import rrt, rrt_star, prm
from amr_path_planner.enhanced_grid import EnhancedGridMap, MovementType
from amr_path_planner.path_smoothing import analyze_path_smoothness


class PerformanceBenchmark:
    """Performance benchmarking suite for path planning algorithms."""
    
    def __init__(self):
        self.results = {}
    
    def create_test_scenarios(self) -> List[Dict]:
        """Create various test scenarios with different complexities."""
        scenarios = []
        
        # Simple scenario
        simple_grid = GridMap(15, 15)
        for i in range(5, 10):
            simple_grid.add_obstacle(i, 7)
        scenarios.append({
            'name': 'Simple',
            'grid': simple_grid,
            'start': (2, 2),
            'goal': (12, 12),
            'description': 'Small grid with simple obstacle'
        })
        
        # Complex maze
        complex_grid = GridMap(25, 25)
        # Create maze-like structure
        for i in range(0, 25, 3):
            for j in range(0, 25, 3):
                if random.random() > 0.3:  # 70% chance of obstacle block
                    for di in range(2):
                        for dj in range(2):
                            if i + di < 25 and j + dj < 25:
                                complex_grid.add_obstacle(i + di, j + dj)
        scenarios.append({
            'name': 'Complex',
            'grid': complex_grid,
            'start': (1, 1),
            'goal': (23, 23),
            'description': 'Large grid with maze-like obstacles'
        })
        
        # Narrow passages
        narrow_grid = GridMap(20, 20)
        # Create narrow corridors
        for i in range(20):
            if i not in [5, 10, 15]:
                narrow_grid.add_obstacle(i, 8)
                narrow_grid.add_obstacle(i, 12)
        for j in range(20):
            if j not in [8, 12]:
                narrow_grid.add_obstacle(10, j)
        scenarios.append({
            'name': 'Narrow',
            'grid': narrow_grid,
            'start': (2, 2),
            'goal': (17, 17),
            'description': 'Grid with narrow passages'
        })
        
        return scenarios
    
    def benchmark_traditional_algorithms(self, scenarios: List[Dict]) -> Dict:
        """Benchmark traditional algorithms (Dijkstra, A*)."""
        print("Benchmarking Traditional Algorithms...")
        results = {}
        
        algorithms_to_test = ['dijkstra', 'astar'] # Corrected 'a_star' to 'astar' if that's the key used in PathPlanner
        
        for scenario in scenarios:
            scenario_name = scenario['name']
            grid = scenario['grid']
            start_pos = scenario['start'] # Renamed
            goal_pos = scenario['goal']   # Renamed
            
            print(f"\n  Testing scenario: {scenario_name}")
            results[scenario_name] = {}
            
            # Initialize PathPlanner once per scenario, algorithm will be set in the loop
            # Or, if PathPlanner's algorithm is fixed at init, create new planner for each.
            # Assuming PathPlanner.plan_path can take an algorithm argument or PathPlanner.change_algorithm exists.
            # The original code uses planner.plan_path(start, goal, algorithm=algorithm)
            # which implies PathPlanner can switch algorithms or takes it as a parameter to plan_path.
            # Let's assume PathPlanner is initialized without a default algorithm, or it can be changed.
            # If PathPlanner requires algorithm at init and cannot change:
            # planner = PathPlanner(grid=grid, algorithm=algorithm_name) # inside the loop
            
            # Based on your PathPlanner, it seems it takes algorithm at init.
            # So, we should create it inside the loop or use change_algorithm.
            # The provided code `planner = PathPlanner(grid)` and then `planner.plan_path(..., algorithm=algorithm)`
            # suggests `plan_path` can specify the algorithm. Let's stick to that if it's true.
            # If not, PathPlanner needs to be initialized with the algorithm.
            # For now, assuming `plan_path` takes an `algorithm` argument.
            # If `PathPlanner` is initialized with an algorithm, then `planner.compute_path(start, goal)` is used.

            # Correcting based on PathPlanner structure:
            # PathPlanner is initialized with an algorithm.
            
            for algorithm_name in algorithms_to_test:
                print(f"    Algorithm: {algorithm_name}")
                # PathPlanner needs to be initialized with the specific algorithm
                planner = PathPlanner(algorithm=algorithm_name, grid=grid)
                times = []
                paths = []
                
                # Run multiple times for statistical significance
                for trial in range(5):
                    start_time_trial = time.perf_counter() # Renamed
                    # Use compute_path as plan_path with algorithm arg might not be standard
                    path = planner.compute_path(start_pos, goal_pos) 
                    end_time_trial = time.perf_counter() # Renamed
                    
                    if path:
                        times.append(end_time_trial - start_time_trial)
                        paths.append(path)
                
                if times:
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    avg_length = np.mean([len(p) for p in paths])
                    
                    path_costs = []
                    for path_segment in paths: # Renamed
                        if not path_segment or len(path_segment) < 2:
                            path_costs.append(float('inf'))
                            continue
                        cost = sum(
                            np.sqrt((path_segment[i+1][0] - path_segment[i][0])**2 + (path_segment[i+1][1] - path_segment[i][1])**2)
                            for i in range(len(path_segment) - 1)
                        )
                        path_costs.append(cost)
                    avg_cost = np.mean(path_costs) if path_costs else 0.0
                    
                    results[scenario_name][algorithm_name] = {
                        'avg_time': avg_time,
                        'std_time': std_time,
                        'avg_length': avg_length,
                        'avg_cost': avg_cost,
                        'success_rate': len(times) / 5,
                        'sample_path': paths[0] if paths else None
                    }
                    
                    print(f"      Time: {avg_time:.4f}±{std_time:.4f}s, Cost: {avg_cost:.2f}")
                else:
                    results[scenario_name][algorithm_name] = {
                        'avg_time': 0.0,
                        'std_time': 0.0,
                        'avg_length': 0.0,
                        'avg_cost': 0.0,
                        'success_rate': 0.0,
                        'sample_path': None
                    }
                    print(f"      Failed to find path")
        
        return results
    
    def benchmark_advanced_algorithms(self, scenarios: List[Dict]) -> Dict:
        """Benchmark advanced algorithms (RRT, RRT*, PRM)."""
        print("\nBenchmarking Advanced Algorithms...")
        results = {}
        
        for scenario in scenarios:
            scenario_name = scenario['name']
            grid = scenario['grid']
            start_pos = scenario['start'] # Renamed to avoid conflict with 'start' time
            goal_pos = scenario['goal']   # Renamed to avoid conflict
            
            print(f"\n  Testing scenario: {scenario_name}")
            results[scenario_name] = {}
            
            # Configure algorithms based on scenario complexity
            if scenario_name == 'Simple':
                iterations = 1000
                samples = 150
            elif scenario_name == 'Complex':
                iterations = 3000
                samples = 400
            else:  # Narrow
                iterations = 2000
                samples = 250
            
            # Define algorithm functions with their parameters
            # These will be called with (start_pos, goal_pos, grid, ...)
            algorithm_runners = {
                'RRT': lambda s, g, gd: rrt(s, g, gd, max_iterations=iterations, step_size=1.5, goal_bias=0.1),
                'RRT*': lambda s, g, gd: rrt_star(s, g, gd, max_iterations=iterations, step_size=1.5, 
                                                  goal_bias=0.1, search_radius=3.0),
                'PRM': lambda s, g, gd: prm(s, g, gd, num_samples=samples, connection_radius=4.0)
            }
            
            for alg_name, alg_func in algorithm_runners.items():
                print(f"    Algorithm: {alg_name}")
                times = []
                paths = []
                
                # Run multiple times for statistical significance
                for trial in range(3):  # Fewer trials due to longer runtime
                    start_time_trial = time.perf_counter() # Renamed to avoid conflict
                    # Call the algorithm function directly
                    path = alg_func(start_pos, goal_pos, grid)
                    end_time_trial = time.perf_counter() # Renamed to avoid conflict
                    
                    if path:
                        times.append(end_time_trial - start_time_trial)
                        paths.append(path)
                
                if times:
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    avg_length = np.mean([len(p) for p in paths])
                    
                    # Calculate path cost
                    path_costs = []
                    for p_idx, path_segment in enumerate(paths): # Renamed path to path_segment
                        if not path_segment or len(path_segment) < 2:
                            # Handle empty or single-point paths if they can occur
                            path_costs.append(float('inf')) # Or 0, or skip
                            continue
                        cost = sum(
                            np.sqrt((path_segment[i+1][0] - path_segment[i][0])**2 + (path_segment[i+1][1] - path_segment[i][1])**2)
                            for i in range(len(path_segment) - 1)
                        )
                        path_costs.append(cost)
                    
                    avg_cost = np.mean(path_costs) if path_costs else 0.0
                    
                    results[scenario_name][alg_name] = {
                        'avg_time': avg_time,
                        'std_time': std_time,
                        'avg_length': avg_length,
                        'avg_cost': avg_cost,
                        'success_rate': len(times) / 3,
                        'sample_path': paths[0] if paths else None
                    }
                    
                    print(f"      Time: {avg_time:.4f}±{std_time:.4f}s, Cost: {avg_cost:.2f}")
                else:
                    results[scenario_name][alg_name] = {
                        'avg_time': 0.0,
                        'std_time': 0.0,
                        'avg_length': 0.0,
                        'avg_cost': 0.0,
                        'success_rate': 0.0,
                        'sample_path': None
                    }
                    print(f"      Failed to find path")
        
        return results
    
    def benchmark_movement_models(self) -> Dict:
        """Benchmark different movement models."""
        print("\nBenchmarking Movement Models...")
        
        # Create test environment
        grid = EnhancedGridMap(20, 20)
        
        # Add obstacles and terrain
        for i in range(6, 14):
            if i not in [9, 10]:
                grid.add_obstacle(i, 10)
        
        # Add terrain costs
        for i in range(20):
            for j in range(20):
                if not grid.is_free(i, j): # Corrected from is_obstacle
                    if 2 <= i <= 5:
                        grid.set_cell_cost(i, j, 3.0) # Corrected from set_terrain_cost
                    elif 15 <= i <= 18:
                        grid.set_cell_cost(i, j, 0.5) # Corrected from set_terrain_cost
        
        start_pos = (2, 2) # Renamed
        goal_pos = (17, 17)  # Renamed
        
        movement_types = [
            MovementType.FOUR_CONNECTED,
            MovementType.EIGHT_CONNECTED,
            MovementType.KNIGHT
        ]
        
        results = {}
        
        for movement_type in movement_types:
            print(f"  Testing {movement_type.value} movement...")
            grid.movement_model = movement_type.value # Corrected from set_movement_type
            
            # Initialize PathPlanner with the algorithm and grid
            planner = PathPlanner(algorithm='astar', grid=grid) # Ensure algorithm is specified
            times = []
            paths = []
            
            for trial in range(5):
                start_time_trial = time.perf_counter() # Renamed
                # Use compute_path method
                path = planner.compute_path(start_pos, goal_pos) 
                end_time_trial = time.perf_counter() # Renamed
                
                if path:
                    times.append(end_time_trial - start_time_trial)
                    paths.append(path)
            
            if times:
                avg_time = np.mean(times)
                avg_length = np.mean([len(p) for p in paths])
                
                # Calculate movement costs
                costs = []
                for path_segment in paths: # Renamed
                    if not path_segment or len(path_segment) < 2:
                        costs.append(float('inf'))
                        continue
                    total_cost = sum(
                        # Ensure get_movement_cost takes two tuples (from_pos, to_pos)
                        grid.get_movement_cost(path_segment[i], path_segment[i+1])
                        for i in range(len(path_segment) - 1)
                    )
                    costs.append(total_cost)
                avg_cost = np.mean(costs) if costs else 0.0
                
                results[movement_type.value] = {
                    'avg_time': avg_time,
                    'avg_length': avg_length,
                    'avg_cost': avg_cost,
                    'success_rate': len(times) / 5,
                    'sample_path': paths[0] if paths else None
                }
                
                print(f"    Time: {avg_time:.4f}s, Cost: {avg_cost:.2f}")
            else:
                results[movement_type.value] = {
                    'avg_time': 0.0, 
                    'avg_length': 0.0, 
                    'avg_cost': 0.0, 
                    'success_rate': 0.0,
                    'sample_path': None
                }
                print(f"    Failed to find path")
        
        return results
    
    def generate_performance_report(self, traditional_results: Dict, 
                                  advanced_results: Dict, movement_results: Dict):
        """Generate comprehensive performance report."""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK REPORT")
        print("="*60)
        
        # Traditional vs Advanced comparison
        print("\n1. ALGORITHM COMPARISON")
        print("-" * 30)
        
        # Assuming scenarios are consistent across traditional_results and advanced_results
        # Get all unique scenario names
        all_scenario_names = set(traditional_results.keys())
        if advanced_results: # Check if advanced_results is not None
            all_scenario_names.update(advanced_results.keys())

        for scenario_name in sorted(list(all_scenario_names)): # Iterate in a defined order
            print(f"\nScenario: {scenario_name}")
            print(f"{'Algorithm':<12} {'Time (s)':<12} {'Cost':<12} {'Success Rate':<12}")
            print("-" * 50)
            
            # Traditional algorithms
            # Ensure 'astar' is used if that's the key in results, not 'a_star'
            for alg in ['dijkstra', 'astar']: # Corrected from 'a_star'
                if scenario_name in traditional_results and alg in traditional_results[scenario_name]:
                    result = traditional_results[scenario_name][alg]
                    if result.get('success_rate', 0) > 0: # Use .get for safety
                        print(f"{alg:<12} {result['avg_time']:.4f} {result['avg_cost']:.2f} {result['success_rate']:.1%}")
                    else:
                        print(f"{alg:<12} {'FAILED':<12} {'N/A':<12} {'0.0%':<12}") # Ensure format consistency
            
            # Advanced algorithms
            for alg in ['RRT', 'RRT*', 'PRM']:
                if scenario_name in advanced_results and alg in advanced_results[scenario_name]:
                    result = advanced_results[scenario_name][alg]
                    if result.get('success_rate', 0) > 0: # Use .get for safety
                        print(f"{alg:<12} {result['avg_time']:.4f} {result['avg_cost']:.2f} {result['success_rate']:.1%}")
                    else:
                        print(f"{alg:<12} {'FAILED':<12} {'N/A':<12} {'0.0%':<12}") # Ensure format consistency
        
        # Movement models comparison
        print("\n2. MOVEMENT MODELS COMPARISON")
        print("-" * 35)
        print(f"{'Movement Type':<20} {'Time (s)':<12} {'Cost':<12} {'Success Rate':<12}")
        print("-" * 60)
        
        if movement_results: # Check if movement_results is not None
            for movement_type, result in movement_results.items():
                if result.get('success_rate', 0) > 0: # Use .get for safety
                    print(f"{movement_type:<20} {result['avg_time']:.4f} {result['avg_cost']:.2f} {result['success_rate']:.1%}")
                else:
                    print(f"{movement_type:<20} {'FAILED':<12} {'N/A':<12} {'0.0%':<12}") # Ensure format consistency
        
        # Key insights
        print("\n3. KEY INSIGHTS")
        print("-" * 15)
        
        insights = [
            "• Traditional algorithms (Dijkstra, A*) are fastest for guaranteed optimal solutions",
            "• RRT* provides better path quality than RRT but takes longer",
            "• PRM excels in complex environments with multiple queries",
            "• 8-connected movement provides more direct paths than 4-connected",
            "• Knight movement can find unique solutions by jumping over obstacles",
            "• Advanced algorithms are probabilistic - success rates may vary"
        ]
        
        for insight in insights:
            print(insight)
    
    def visualize_performance_comparison(self, traditional_results: Dict,
                                       advanced_results: Dict, movement_results: Dict):
        """Create performance visualization charts."""
        try:
            # Time comparison chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 13)) # Adjusted size
            fig.suptitle('Performance Benchmark Results', fontsize=16)

            # 1. Algorithm execution time comparison
            all_scenario_names = set()
            if traditional_results:
                all_scenario_names.update(traditional_results.keys())
            if advanced_results:
                all_scenario_names.update(advanced_results.keys())
            
            scenarios_list = sorted(list(all_scenario_names))
            # Ensure 'astar' is used if that's the key in results
            algorithms_list = ['dijkstra', 'astar', 'RRT', 'RRT*', 'PRM'] 

            times_data = []
            labels = []

            for scenario_name in scenarios_list:
                scenario_times = []
                for alg_name in algorithms_list:
                    time_val = 0 # Default to 0 if not found or failed
                    if traditional_results and scenario_name in traditional_results and alg_name in traditional_results[scenario_name]:
                        res = traditional_results[scenario_name][alg_name]
                        if res.get('success_rate', 0) > 0:
                            time_val = res['avg_time']
                    elif advanced_results and scenario_name in advanced_results and alg_name in advanced_results[scenario_name]:
                        res = advanced_results[scenario_name][alg_name]
                        if res.get('success_rate', 0) > 0:
                            time_val = res['avg_time']
                    scenario_times.append(time_val)
                
                if any(t > 0 for t in scenario_times): # Only add if there's data for this scenario
                    times_data.append(scenario_times)
                    labels.append(scenario_name)
            
            if times_data: # Proceed only if there's data to plot
                x = np.arange(len(algorithms_list))
                num_scenarios = len(times_data)
                # Adjust width based on number of scenarios to prevent overlap
                total_width_for_bars = 0.8 
                width = total_width_for_bars / num_scenarios if num_scenarios > 0 else 0.25

                for i, (times, label) in enumerate(zip(times_data, labels)):
                    # Offset bars for each scenario
                    ax1.bar(x - total_width_for_bars/2 + i * width + width/2, times, width, label=label)
                
                ax1.set_xlabel('Algorithms')
                ax1.set_ylabel('Execution Time (s)')
                ax1.set_title('Algorithm Execution Time Comparison')
                ax1.set_xticks(x)
                ax1.set_xticklabels(algorithms_list, rotation=45, ha="right")
                ax1.legend(title="Scenarios")
                ax1.set_yscale('log') # Keep log scale if times vary widely
                ax1.grid(True, which="both", ls="--", alpha=0.7)
            else:
                ax1.text(0.5, 0.5, 'No successful runs for time comparison', ha='center', va='center', transform=ax1.transAxes)


            # 2. Path cost comparison
            costs_data = []
            # labels are already defined from the time comparison part
            
            for scenario_name in scenarios_list: # Use the same scenarios_list for consistency
                scenario_costs = []
                for alg_name in algorithms_list:
                    cost_val = 0 # Default to 0 or np.nan if preferred for missing data
                    if traditional_results and scenario_name in traditional_results and alg_name in traditional_results[scenario_name]:
                        res = traditional_results[scenario_name][alg_name]
                        if res.get('success_rate', 0) > 0:
                            cost_val = res['avg_cost']
                    elif advanced_results and scenario_name in advanced_results and alg_name in advanced_results[scenario_name]:
                        res = advanced_results[scenario_name][alg_name]
                        if res.get('success_rate', 0) > 0:
                            cost_val = res['avg_cost']
                    scenario_costs.append(cost_val)
                
                # Only add if there's data for this scenario (matching times_data logic)
                if any(c > 0 for c in scenario_costs) and scenario_name in labels: 
                    costs_data.append(scenario_costs)

            if costs_data: # Proceed only if there's data to plot
                # x and width are already defined
                for i, (costs, label) in enumerate(zip(costs_data, labels)): # Use the same labels
                    ax2.bar(x - total_width_for_bars/2 + i * width + width/2, costs, width, label=label)
                
                ax2.set_xlabel('Algorithms')
                ax2.set_ylabel('Path Cost')
                ax2.set_title('Path Cost Comparison')
                ax2.set_xticks(x)
                ax2.set_xticklabels(algorithms_list, rotation=45, ha="right")
                ax2.legend(title="Scenarios")
                ax2.grid(True, which="both", ls="--", alpha=0.7)
            else:
                ax2.text(0.5, 0.5, 'No successful runs for cost comparison', ha='center', va='center', transform=ax2.transAxes)

            # 3. Movement models comparison
            if movement_results: # Check if movement_results is not None and not empty
                movement_names = list(movement_results.keys())
                # Filter out entries with 0 success rate before plotting
                valid_movement_names = [name for name in movement_names if movement_results[name].get('success_rate', 0) > 0]
                
                if valid_movement_names:
                    movement_times = [movement_results[name]['avg_time'] for name in valid_movement_names]
                    movement_costs = [movement_results[name]['avg_cost'] for name in valid_movement_names]
                
                    ax3.bar(valid_movement_names, movement_times, color='skyblue')
                    ax3.set_xlabel('Movement Type')
                    ax3.set_ylabel('Execution Time (s)')
                    ax3.set_title('Movement Models - Execution Time')
                    ax3.tick_params(axis='x', rotation=45) # Removed ha="right"
                    ax3.grid(True, which="both", ls="--", alpha=0.7)

                    ax4.bar(valid_movement_names, movement_costs, color='lightcoral')
                    ax4.set_xlabel('Movement Type')
                    ax4.set_ylabel('Path Cost')
                    ax4.set_title('Movement Models - Path Cost')
                    ax4.tick_params(axis='x', rotation=45) # Removed ha="right"
                    ax4.grid(True, which="both", ls="--", alpha=0.7)
                else:
                    ax3.text(0.5, 0.5, 'No successful runs for movement time', ha='center', va='center', transform=ax3.transAxes)
                    ax4.text(0.5, 0.5, 'No successful runs for movement cost', ha='center', va='center', transform=ax4.transAxes)
            else:
                ax3.text(0.5, 0.5, 'No movement model data', ha='center', va='center', transform=ax3.transAxes)
                ax4.text(0.5, 0.5, 'No movement model data', ha='center', va='center', transform=ax4.transAxes)

            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
            plt.savefig('performance_benchmark.png', dpi=150, bbox_inches='tight')
            print("\nPerformance comparison chart saved as performance_benchmark.png")
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()

    def run_full_benchmark(self):
        """Run complete performance benchmark."""
        print("Starting Performance Benchmark Suite...")
        
        # Create test scenarios
        scenarios = self.create_test_scenarios()
        print(f"Created {len(scenarios)} test scenarios")
        
        # Run benchmarks
        traditional_results = self.benchmark_traditional_algorithms(scenarios)
        advanced_results = self.benchmark_advanced_algorithms(scenarios)
        movement_results = self.benchmark_movement_models()
        
        # Generate report
        self.generate_performance_report(traditional_results, advanced_results, movement_results)
        
        # Create visualizations
        self.visualize_performance_comparison(traditional_results, advanced_results, movement_results)
        
        return {
            'traditional': traditional_results,
            'advanced': advanced_results,
            'movement': movement_results
        }

# This should be outside the class definition
def main():
    """Run the performance benchmark."""
    try:
        benchmark = PerformanceBenchmark()
        results = benchmark.run_full_benchmark()
        
        print(f"\nBenchmark completed! Results saved to performance_benchmark.png")
        
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Please install required packages: pip install matplotlib numpy")
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
