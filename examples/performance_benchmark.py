"""
Performance comparison demo between traditional and advanced algorithms.

This script benchmarks different algorithms and provides detailed performance analysis.

Updates in this version (aligned with your improved codebase):
- Uses PathPlanner.compute_path() consistently (no plan_path / algorithm arg confusion).
- Fixes terrain-cost setup bug (was applying costs to obstacles instead of free cells).
- Handles EnhancedGrid movement model as string values (keeps MovementType if present).
- Adds optional smoothing benchmark using smooth_path + analyze_path_smoothness.
- Makes scenario generation more robust (guarantees start/goal are free; regenerates if needed).
- Produces cleaner plots even when some algorithms fail (no crashes, no wrong indexing).
"""

from __future__ import annotations

import os
import random
import sys
import time
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_path_planner import PathPlanner, GridMap
from amr_path_planner.advanced_algorithms import rrt, rrt_star, prm
from amr_path_planner.enhanced_grid import EnhancedGridMap

# MovementType may or may not exist in your package; keep it optional.
try:
    from amr_path_planner.enhanced_grid import MovementType  # type: ignore
except Exception:
    MovementType = None  # type: ignore

from amr_path_planner.path_smoothing import analyze_path_smoothness, smooth_path


Point = Tuple[int, int]


def _euclidean_path_cost(path: List[Point]) -> float:
    if not path or len(path) < 2:
        return 0.0
    return float(
        sum(np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))
    )


def _ensure_free(grid: GridMap, p: Point) -> bool:
    return grid.is_free(p[0], p[1])


def _regen_complex_maze(seed: int, w: int = 25, h: int = 25) -> GridMap:
    rng = random.Random(seed)
    g = GridMap(w, h)
    for i in range(0, w, 3):
        for j in range(0, h, 3):
            if rng.random() > 0.3:  # 70% chance of obstacle block
                for di in range(2):
                    for dj in range(2):
                        x, y = i + di, j + dj
                        if 0 <= x < w and 0 <= y < h:
                            g.add_obstacle(x, y)
    return g


class PerformanceBenchmark:
    """Performance benchmarking suite for path planning algorithms."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def create_test_scenarios(self) -> List[Dict]:
        """Create various test scenarios with different complexities."""
        scenarios: List[Dict] = []

        # 1) Simple scenario
        simple_grid = GridMap(15, 15)
        for i in range(5, 10):
            simple_grid.add_obstacle(i, 7)
        scenarios.append(
            {
                "name": "Simple",
                "grid": simple_grid,
                "start": (2, 2),
                "goal": (12, 12),
                "description": "Small grid with simple obstacle",
            }
        )

        # 2) Complex maze (regenerate until start/goal are free)
        start_c: Point = (1, 1)
        goal_c: Point = (23, 23)
        complex_grid = None
        for attempt in range(50):
            complex_grid = _regen_complex_maze(seed=self.seed + attempt, w=25, h=25)
            if _ensure_free(complex_grid, start_c) and _ensure_free(complex_grid, goal_c):
                break
        if complex_grid is None:
            complex_grid = _regen_complex_maze(seed=self.seed, w=25, h=25)
        scenarios.append(
            {
                "name": "Complex",
                "grid": complex_grid,
                "start": start_c,
                "goal": goal_c,
                "description": "Large grid with maze-like obstacles",
            }
        )

        # 3) Narrow passages
        narrow_grid = GridMap(20, 20)
        for i in range(20):
            if i not in [5, 10, 15]:
                narrow_grid.add_obstacle(i, 8)
                narrow_grid.add_obstacle(i, 12)
        for j in range(20):
            if j not in [8, 12]:
                narrow_grid.add_obstacle(10, j)

        # Ensure start/goal are free (if not, adjust)
        start_n: Point = (2, 2)
        goal_n: Point = (17, 17)
        if not _ensure_free(narrow_grid, start_n):
            start_n = (1, 1)
        if not _ensure_free(narrow_grid, goal_n):
            goal_n = (18, 18)

        scenarios.append(
            {
                "name": "Narrow",
                "grid": narrow_grid,
                "start": start_n,
                "goal": goal_n,
                "description": "Grid with narrow passages",
            }
        )

        return scenarios

    # ----------------------------
    # Traditional algorithms
    # ----------------------------

    def benchmark_traditional_algorithms(self, scenarios: List[Dict]) -> Dict:
        """Benchmark traditional algorithms (Dijkstra, A*)."""
        print("Benchmarking Traditional Algorithms...")
        results: Dict[str, Dict[str, dict]] = {}

        algorithms = ["dijkstra", "astar"]

        for scenario in scenarios:
            scenario_name = scenario["name"]
            grid: GridMap = scenario["grid"]
            start_pos: Point = scenario["start"]
            goal_pos: Point = scenario["goal"]

            print(f"\n  Testing scenario: {scenario_name}")
            results[scenario_name] = {}

            for alg in algorithms:
                print(f"    Algorithm: {alg}")
                planner = PathPlanner(algorithm=alg, grid=grid)

                times: List[float] = []
                paths: List[List[Point]] = []

                for _trial in range(5):
                    t0 = time.perf_counter()
                    path = planner.compute_path(start_pos, goal_pos)
                    t1 = time.perf_counter()

                    if path:
                        times.append(t1 - t0)
                        paths.append(path)

                if times:
                    avg_time = float(np.mean(times))
                    std_time = float(np.std(times))
                    avg_len = float(np.mean([len(p) for p in paths]))
                    avg_cost = float(np.mean([_euclidean_path_cost(p) for p in paths]))

                    results[scenario_name][alg] = {
                        "avg_time": avg_time,
                        "std_time": std_time,
                        "avg_length": avg_len,
                        "avg_cost": avg_cost,
                        "success_rate": len(times) / 5.0,
                        "sample_path": paths[0],
                    }
                    print(f"      Time: {avg_time:.4f}±{std_time:.4f}s, Cost: {avg_cost:.2f}")
                else:
                    results[scenario_name][alg] = {
                        "avg_time": 0.0,
                        "std_time": 0.0,
                        "avg_length": 0.0,
                        "avg_cost": 0.0,
                        "success_rate": 0.0,
                        "sample_path": None,
                    }
                    print("      Failed to find path")

        return results

    # ----------------------------
    # Advanced algorithms
    # ----------------------------

    def benchmark_advanced_algorithms(self, scenarios: List[Dict]) -> Dict:
        """Benchmark advanced algorithms (RRT, RRT*, PRM)."""
        print("\nBenchmarking Advanced Algorithms...")
        results: Dict[str, Dict[str, dict]] = {}

        for scenario in scenarios:
            scenario_name = scenario["name"]
            grid: GridMap = scenario["grid"]
            start_pos: Point = scenario["start"]
            goal_pos: Point = scenario["goal"]

            print(f"\n  Testing scenario: {scenario_name}")
            results[scenario_name] = {}

            # Scenario-based parameterization
            if scenario_name == "Simple":
                iterations = 1000
                samples = 150
            elif scenario_name == "Complex":
                iterations = 3000
                samples = 400
            else:  # Narrow
                iterations = 2000
                samples = 250

            runners = {
                "RRT": lambda s, g, gd: rrt(s, g, gd, max_iterations=iterations, step_size=1.5, goal_bias=0.1),
                "RRT*": lambda s, g, gd: rrt_star(
                    s, g, gd, max_iterations=iterations, step_size=1.5, goal_bias=0.1, search_radius=3.0
                ),
                "PRM": lambda s, g, gd: prm(s, g, gd, num_samples=samples, connection_radius=4.0),
            }

            for name, fn in runners.items():
                print(f"    Algorithm: {name}")
                times: List[float] = []
                paths: List[List[Point]] = []

                # fewer trials for sampling planners
                for _trial in range(3):
                    t0 = time.perf_counter()
                    path = fn(start_pos, goal_pos, grid)
                    t1 = time.perf_counter()

                    if path:
                        times.append(t1 - t0)
                        paths.append(path)

                if times:
                    avg_time = float(np.mean(times))
                    std_time = float(np.std(times))
                    avg_len = float(np.mean([len(p) for p in paths]))
                    avg_cost = float(np.mean([_euclidean_path_cost(p) for p in paths]))

                    results[scenario_name][name] = {
                        "avg_time": avg_time,
                        "std_time": std_time,
                        "avg_length": avg_len,
                        "avg_cost": avg_cost,
                        "success_rate": len(times) / 3.0,
                        "sample_path": paths[0],
                    }
                    print(f"      Time: {avg_time:.4f}±{std_time:.4f}s, Cost: {avg_cost:.2f}")
                else:
                    results[scenario_name][name] = {
                        "avg_time": 0.0,
                        "std_time": 0.0,
                        "avg_length": 0.0,
                        "avg_cost": 0.0,
                        "success_rate": 0.0,
                        "sample_path": None,
                    }
                    print("      Failed to find path")

        return results

    # ----------------------------
    # Movement models
    # ----------------------------

    def benchmark_movement_models(self) -> Dict:
        """Benchmark different movement models on EnhancedGridMap."""
        print("\nBenchmarking Movement Models...")

        grid = EnhancedGridMap(20, 20)

        # Obstacles
        for i in range(6, 14):
            if i not in [9, 10]:
                grid.add_obstacle(i, 10)

        # Terrain costs (FIXED: apply to free cells, not obstacles)
        for i in range(grid.width):
            for j in range(grid.height):
                if grid.is_free(i, j):
                    if 2 <= i <= 5:
                        grid.set_cell_cost(i, j, 3.0)   # difficult
                    elif 15 <= i <= 18:
                        grid.set_cell_cost(i, j, 0.5)   # easy

        start_pos: Point = (2, 2)
        goal_pos: Point = (17, 17)

        # Movement types: support enum if present, else fallback strings
        if MovementType is not None:
            movement_names = [
                MovementType.FOUR_CONNECTED.value,
                MovementType.EIGHT_CONNECTED.value,
                MovementType.KNIGHT.value,
            ]
        else:
            movement_names = ["4-connected", "8-connected", "knight"]

        results: Dict[str, dict] = {}

        for m in movement_names:
            print(f"  Testing {m} movement...")
            grid.movement_model = m

            planner = PathPlanner(algorithm="astar", grid=grid)
            times: List[float] = []
            paths: List[List[Point]] = []

            for _trial in range(5):
                t0 = time.perf_counter()
                path = planner.compute_path(start_pos, goal_pos)
                t1 = time.perf_counter()

                if path:
                    times.append(t1 - t0)
                    paths.append(path)

            if times:
                avg_time = float(np.mean(times))
                avg_len = float(np.mean([len(p) for p in paths]))
                avg_cost = float(np.mean([sum(grid.get_movement_cost(p[i], p[i + 1]) for i in range(len(p) - 1)) for p in paths]))

                results[m] = {
                    "avg_time": avg_time,
                    "avg_length": avg_len,
                    "avg_cost": avg_cost,
                    "success_rate": len(times) / 5.0,
                    "sample_path": paths[0],
                }
                print(f"    Time: {avg_time:.4f}s, Cost: {avg_cost:.2f}")
            else:
                results[m] = {
                    "avg_time": 0.0,
                    "avg_length": 0.0,
                    "avg_cost": 0.0,
                    "success_rate": 0.0,
                    "sample_path": None,
                }
                print("    Failed to find path")

        return results

    # ----------------------------
    # Optional smoothing benchmark
    # ----------------------------

    def benchmark_smoothing(self, scenarios: List[Dict]) -> Dict:
        """Benchmark smoothing (compute A* path then smooth it)."""
        print("\nBenchmarking Path Smoothing...")
        results: Dict[str, dict] = {}

        for scenario in scenarios:
            name = scenario["name"]
            grid: GridMap = scenario["grid"]
            start_pos: Point = scenario["start"]
            goal_pos: Point = scenario["goal"]

            planner = PathPlanner(algorithm="astar", grid=grid)

            t0 = time.perf_counter()
            path = planner.compute_path(start_pos, goal_pos)
            t1 = time.perf_counter()

            if not path:
                results[name] = {"success": False}
                print(f"  {name}: base path FAILED")
                continue

            base_time = t1 - t0
            base_cost = _euclidean_path_cost(path)
            base_smooth = analyze_path_smoothness(path)

            # Smooth
            t2 = time.perf_counter()
            sm = smooth_path(path, grid, method="shortcut", max_iterations=200)
            t3 = time.perf_counter()

            sm_time = t3 - t2
            sm_cost = _euclidean_path_cost(sm)
            sm_smooth = analyze_path_smoothness(sm)

            results[name] = {
                "success": True,
                "base_time": float(base_time),
                "smooth_time": float(sm_time),
                "base_cost": float(base_cost),
                "smooth_cost": float(sm_cost),
                "base_smoothness": base_smooth,
                "smooth_smoothness": sm_smooth,
                "base_len": len(path),
                "smooth_len": len(sm),
            }

            print(
                f"  {name}: base_time={base_time:.4f}s, smooth_time={sm_time:.4f}s, "
                f"cost {base_cost:.2f}->{sm_cost:.2f}, points {len(path)}->{len(sm)}"
            )

        return results

    # ----------------------------
    # Reporting
    # ----------------------------

    def generate_performance_report(
        self,
        traditional_results: Dict,
        advanced_results: Dict,
        movement_results: Dict,
        smoothing_results: Optional[Dict] = None,
    ) -> None:
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK REPORT")
        print("=" * 60)

        print("\n1. ALGORITHM COMPARISON")
        print("-" * 30)

        all_scenarios = set(traditional_results.keys()) | set(advanced_results.keys())

        for scenario_name in sorted(all_scenarios):
            print(f"\nScenario: {scenario_name}")
            print(f"{'Algorithm':<12} {'Time (s)':<12} {'Cost':<12} {'Success':<10}")
            print("-" * 50)

            for alg in ["dijkstra", "astar"]:
                r = traditional_results.get(scenario_name, {}).get(alg)
                if r and r.get("success_rate", 0) > 0:
                    print(f"{alg:<12} {r['avg_time']:<12.4f} {r['avg_cost']:<12.2f} {r['success_rate']:<10.1%}")
                else:
                    print(f"{alg:<12} {'FAILED':<12} {'N/A':<12} {'0.0%':<10}")

            for alg in ["RRT", "RRT*", "PRM"]:
                r = advanced_results.get(scenario_name, {}).get(alg)
                if r and r.get("success_rate", 0) > 0:
                    print(f"{alg:<12} {r['avg_time']:<12.4f} {r['avg_cost']:<12.2f} {r['success_rate']:<10.1%}")
                else:
                    print(f"{alg:<12} {'FAILED':<12} {'N/A':<12} {'0.0%':<10}")

        print("\n2. MOVEMENT MODELS COMPARISON")
        print("-" * 35)
        print(f"{'Movement Type':<20} {'Time (s)':<12} {'Cost':<12} {'Success':<10}")
        print("-" * 60)
        for m, r in movement_results.items():
            if r.get("success_rate", 0) > 0:
                print(f"{m:<20} {r['avg_time']:<12.4f} {r['avg_cost']:<12.2f} {r['success_rate']:<10.1%}")
            else:
                print(f"{m:<20} {'FAILED':<12} {'N/A':<12} {'0.0%':<10}")

        if smoothing_results:
            print("\n3. SMOOTHING (A* base + Shortcut smoothing)")
            print("-" * 45)
            print(f"{'Scenario':<10} {'Base(s)':<10} {'Smooth(s)':<10} {'Cost->':<18} {'Pts->':<14}")
            print("-" * 65)
            for sname, r in smoothing_results.items():
                if not r.get("success"):
                    print(f"{sname:<10} {'FAILED':<10}")
                    continue
                print(
                    f"{sname:<10} {r['base_time']:<10.4f} {r['smooth_time']:<10.4f} "
                    f"{r['base_cost']:.2f}->{r['smooth_cost']:.2f}   {r['base_len']}->{r['smooth_len']}"
                )

        print("\n4. KEY INSIGHTS")
        print("-" * 15)
        for line in [
            "• Dijkstra/A* are deterministic; A* is typically faster with a good heuristic.",
            "• RRT*/PRM are probabilistic; success rate and runtime vary with samples/iterations.",
            "• RRT* improves path quality vs RRT at additional compute cost.",
            "• PRM can shine in complex maps (especially for repeated queries) but needs enough samples.",
            "• 8-connected movement yields more direct routes than 4-connected in open terrain.",
            "• Terrain costs can trade distance for cheaper traversal when using EnhancedGridMap.",
            "• Smoothing can reduce waypoints and turning, but must remain collision-safe.",
        ]:
            print(line)

    def visualize_performance_comparison(
        self,
        traditional_results: Dict,
        advanced_results: Dict,
        movement_results: Dict,
    ) -> None:
        """Create performance visualization charts."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 13))
            fig.suptitle("Performance Benchmark Results", fontsize=16)

            scenarios = sorted(set(traditional_results.keys()) | set(advanced_results.keys()))
            algorithms = ["dijkstra", "astar", "RRT", "RRT*", "PRM"]

            # Helper to fetch metric
            def get_metric(sname: str, alg: str, metric: str) -> float:
                if alg in ("dijkstra", "astar"):
                    r = traditional_results.get(sname, {}).get(alg)
                else:
                    r = advanced_results.get(sname, {}).get(alg)
                if r and r.get("success_rate", 0) > 0:
                    return float(r.get(metric, 0.0))
                return 0.0

            # 1) Times (log-scale)
            times_data = [[get_metric(s, a, "avg_time") for a in algorithms] for s in scenarios]
            x = np.arange(len(algorithms))
            total_width = 0.8
            width = total_width / max(1, len(scenarios))

            for i, sname in enumerate(scenarios):
                ax1.bar(x - total_width / 2 + i * width + width / 2, times_data[i], width, label=sname)

            ax1.set_xlabel("Algorithms")
            ax1.set_ylabel("Execution Time (s)")
            ax1.set_title("Algorithm Execution Time Comparison")
            ax1.set_xticks(x)
            ax1.set_xticklabels(algorithms, rotation=45, ha="right")
            ax1.set_yscale("log")
            ax1.grid(True, which="both", ls="--", alpha=0.7)
            ax1.legend(title="Scenarios")

            # 2) Costs
            costs_data = [[get_metric(s, a, "avg_cost") for a in algorithms] for s in scenarios]
            for i, sname in enumerate(scenarios):
                ax2.bar(x - total_width / 2 + i * width + width / 2, costs_data[i], width, label=sname)

            ax2.set_xlabel("Algorithms")
            ax2.set_ylabel("Path Cost")
            ax2.set_title("Path Cost Comparison")
            ax2.set_xticks(x)
            ax2.set_xticklabels(algorithms, rotation=45, ha="right")
            ax2.grid(True, which="both", ls="--", alpha=0.7)
            ax2.legend(title="Scenarios")

            # 3) Movement times
            mv_names = [k for k, v in movement_results.items() if v.get("success_rate", 0) > 0]
            if mv_names:
                mv_times = [movement_results[k]["avg_time"] for k in mv_names]
                ax3.bar(mv_names, mv_times)
                ax3.set_xlabel("Movement Type")
                ax3.set_ylabel("Execution Time (s)")
                ax3.set_title("Movement Models - Execution Time")
                ax3.tick_params(axis="x", rotation=45)
                ax3.grid(True, which="both", ls="--", alpha=0.7)
            else:
                ax3.text(0.5, 0.5, "No successful movement runs", ha="center", va="center", transform=ax3.transAxes)

            # 4) Movement costs
            if mv_names:
                mv_costs = [movement_results[k]["avg_cost"] for k in mv_names]
                ax4.bar(mv_names, mv_costs)
                ax4.set_xlabel("Movement Type")
                ax4.set_ylabel("Path Cost")
                ax4.set_title("Movement Models - Path Cost")
                ax4.tick_params(axis="x", rotation=45)
                ax4.grid(True, which="both", ls="--", alpha=0.7)
            else:
                ax4.text(0.5, 0.5, "No successful movement runs", ha="center", va="center", transform=ax4.transAxes)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig("performance_benchmark.png", dpi=150, bbox_inches="tight")
            print("\nPerformance comparison chart saved as performance_benchmark.png")
            plt.show()

        except Exception as e:
            print(f"Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()

    def run_full_benchmark(self) -> Dict:
        """Run complete performance benchmark."""
        print("Starting Performance Benchmark Suite...")

        scenarios = self.create_test_scenarios()
        print(f"Created {len(scenarios)} test scenarios")

        traditional_results = self.benchmark_traditional_algorithms(scenarios)
        advanced_results = self.benchmark_advanced_algorithms(scenarios)
        movement_results = self.benchmark_movement_models()
        smoothing_results = self.benchmark_smoothing(scenarios)

        self.generate_performance_report(traditional_results, advanced_results, movement_results, smoothing_results)
        self.visualize_performance_comparison(traditional_results, advanced_results, movement_results)

        return {
            "traditional": traditional_results,
            "advanced": advanced_results,
            "movement": movement_results,
            "smoothing": smoothing_results,
        }


def main() -> None:
    """Run the performance benchmark."""
    try:
        benchmark = PerformanceBenchmark(seed=42)
        _results = benchmark.run_full_benchmark()
        print("\nBenchmark completed! Results saved to performance_benchmark.png")

    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Please install required packages: pip install matplotlib numpy")
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
