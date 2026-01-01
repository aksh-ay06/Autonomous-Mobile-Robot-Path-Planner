# Autonomous Mobile Robot Path Planner

Lightweight Python toolkit for grid-based motion planning with dynamic obstacles, smoothing, and optional sampling-based planners (RRT, RRT*, PRM). Works for single-robot and multi-robot scenarios with space-time reservations.

## Features
- Grid planning: Dijkstra and A* with pluggable heuristics and optional enhanced grids (8-connected, custom moves, terrain costs).
- Sampling-based planners: RRT, RRT*, PRM with line-of-sight collision checks.
- Path smoothing: shortcut, adaptive, Douglas–Peucker, Chaikin-based "bezier", and spline (SciPy optional).
- Dynamic obstacles: random-walk obstacle manager with collision-aware planning.
- Multi-robot coordination: space-time A* with vertex/edge reservations and priority/cooperative/centralized modes.
- Visualization: matplotlib-based simulator for single or multi-robot runs (optional GIF export).

## Installation
```bash
python3 -m venv .amr-env
source .amr-env/bin/activate
pip install --upgrade pip
pip install -e .
```
Python 3.8+ is required. For spline smoothing with Gaussian filtering, install SciPy:
```bash
pip install scipy
```

## Quick start
### Single robot
```python
from amr_path_planner import GridMap, PathPlanner, RobotAgent

grid = GridMap(width=20, height=20, static_obstacles={(5, 5), (6, 5)})
planner = PathPlanner(algorithm="astar", grid=grid)
robot = RobotAgent(position=(0, 0), planner=planner)

robot.plan_to((15, 15))
while not robot.is_at_goal():
    robot.step()
print("Reached", robot.position)
```

### With dynamic obstacles
```python
from amr_path_planner import DynamicObstacleMgr

obstacles = DynamicObstacleMgr(grid)
obstacles.spawn_random_obstacles(15, max_attempts=200)
robot.plan_to((15, 15), obstacle_mgr=obstacles)
while not robot.is_at_goal():
    obstacles.update()
    robot.step(obstacle_mgr=obstacles)
```

### Multi-robot coordination
```python
from amr_path_planner import MultiRobotCoordinator, PathPlanner, RobotAgent, GridMap

grid = GridMap(20, 20)
coord = MultiRobotCoordinator(grid, coordination_mode="priority", horizon=15)

for start, goal in [((0, 0), (10, 10)), ((19, 0), (10, 9))]:
    planner = PathPlanner(grid=grid)
    robot = RobotAgent(position=start, planner=planner)
    rid = coord.add_robot(robot)
    coord.set_robot_goal(rid, goal)

coord.plan_coordinated_paths()
for _ in range(30):
    coord.step_all_robots()
    if all(r.is_at_goal() for r in coord.robots):
        break
```

### Smoothing a path
```python
from amr_path_planner import smooth_path, GridMap

grid = GridMap(10, 10)
raw = [(0,0),(1,0),(1,1),(2,1),(3,1),(4,2),(5,2)]
smoothed = smooth_path(raw, grid, method="shortcut", max_iterations=200)
```

## Examples
See the `examples/` directory:
- `demo.py`: basic single-robot planning.
- `advanced_features_demo.py`: enhanced grid, smoothing, and sampling planners.
- `performance_benchmark.py`: quick perf checks.
- `demo_single_robot.py` at repo root: simple visualization entry point.
- `multi_demo.py` at repo root: configurable multi-robot + dynamic obstacle demo. Example:
    ```bash
    python3 multi_demo.py \
        --width 15 --height 15 \
        --robots 5 \
        --obstacles 50 \
        --dynamic-obstacles 8 \
        --dynamic-move-prob 0.7 \
        --horizon 80 \
        --mode cooperative \
        --step-delay 0.05
    ```
    Flags:
    - `--width/--height`: grid size
    - `--obstacles`: static obstacle count
    - `--dynamic-obstacles`: moving obstacles to spawn
    - `--dynamic-move-prob`: move probability per step for dynamic obstacles
    - `--robots`: number of robots; start/goal pairs are sampled randomly (seeded)
    - `--horizon`: space-time planning horizon
    - `--mode`: `priority`, `cooperative`, or `centralized`
    - `--step-delay`: visualization delay (seconds)
    - `--max-steps`: simulation cap

## Running tests
Use the bundled virtual environment or your own:
```bash
pytest
```

## Key modules (overview)
- `amr_path_planner/grid_map.py`: base grid with static obstacles.
- `amr_path_planner/enhanced_grid.py`: movement models (4/8/custom) and terrain costs.
- `amr_path_planner/search_algorithms.py`: A*, Dijkstra, space-time A*.
- `amr_path_planner/advanced_algorithms.py`: RRT, RRT*, PRM.
- `amr_path_planner/path_planner.py`: unified planner interface + smoothing.
- `amr_path_planner/path_smoothing.py`: smoothing algorithms and metrics.
- `amr_path_planner/dynamic_obstacles.py`: random-walk dynamic obstacle manager.
- `amr_path_planner/multi_robot_coordinator.py`: reservation-based multi-robot planner.
- `amr_path_planner/simulator.py`: matplotlib visualization.

## Notes
- Sampling-based planners are probabilistic; you can pass seeds for reproducibility.
- Space-time coordination uses a finite horizon; increase `horizon` for longer plans.
- Smoothing is collision-aware when given a grid; for pure geometry, pass `collision_safe=False` to `smooth_path` if you need unbounded simplification.
