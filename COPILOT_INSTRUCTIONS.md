# GitHub Copilot Instruction File
_**Autonomous Mobile Robot Path Planner**_

Use these instructions in GitHub Copilot to scaffold and implement an end-to-end Python project. Copy the contents into a file named `COPILOT_INSTRUCTIONS.md` or paste at the top of your main module.

---

## 1. Project Initialization

1. **Generate repository structure**:
   ```text
   amr_path_planner/
   ├── amr_path_planner/        # Package directory
   │   ├── __init__.py
   │   ├── grid_map.py          # GridMap class
   │   ├── search_algorithms.py # Dijkstra & A* implementations
   │   ├── path_planner.py      # PathPlanner wrapper
   │   ├── dynamic_obstacles.py # DynamicObstacleMgr
   │   ├── robot_agent.py       # RobotAgent logic
   │   └── simulator.py         # Simulation loop + visualization
   ├── tests/                   # Unit tests
   │   ├── test_grid_map.py
   │   ├── test_search_algorithms.py
   │   └── test_path_planner.py
   ├── examples/                # Example scripts or notebooks
   │   └── demo.py
   ├── requirements.txt
   ├── README.md
   └── COPILOT_INSTRUCTIONS.md  # (this file)
   ```

2. **Generate `requirements.txt`** listing: `matplotlib`, `networkx` (optional), `pytest`.

---

## 2. GridMap Module (`grid_map.py`)

- **Class**: `GridMap`
  - **Constructor**: takes `width: int`, `height: int`, and optional `static_obstacles: Set[Tuple[int,int]]`
  - **Methods**:
    - `is_free(x: int, y: int) -> bool`
    - `neighbors(x: int, y: int) -> List[Tuple[int,int]]`

> *Copilot prompt:* "Generate `GridMap` class with methods `is_free` and `neighbors` for a 4-connected grid."  

---

## 3. Search Algorithms (`search_algorithms.py`)

- **Functions**:
  - `dijkstra(start: Tuple[int,int], goal: Tuple[int,int], grid: GridMap) -> List[Tuple[int,int]]`
  - `astar(start: Tuple[int,int], goal: Tuple[int,int], grid: GridMap, heuristic: Callable) -> List[Tuple[int,int]]`
- Use a **priority queue** for the frontier.
- Manhattan distance heuristic for A*.

> *Copilot prompt:* "Implement Dijkstra and A* search algorithms returning path as list of coordinates."  

---

## 4. Path Planner Wrapper (`path_planner.py`)

- **Class**: `PathPlanner`
  - **Constructor**: takes algorithm name (`'dijkstra'` or `'astar'`) and optional heuristic
  - **Method**: `compute_path(start, goal) -> List[Tuple[int,int]]`

> *Copilot prompt:* "Create `PathPlanner` class that wraps search functions and exposes `compute_path` method."  

---

## 5. Dynamic Obstacles (`dynamic_obstacles.py`)

- **Class**: `DynamicObstacleMgr`
  - Maintains a list of obstacle positions and movement rules
  - **Methods**:
    - `update()` advances all obstacles one step
    - `is_collision(x, y) -> bool`

> *Copilot prompt:* "Generate `DynamicObstacleMgr` with random-walk obstacles and collision check."  

---

## 6. Robot Agent (`robot_agent.py`)

- **Class**: `RobotAgent`
  - Tracks `position`, `path`, and `planner`
  - **Methods**:
    - `plan_to(goal: Tuple[int,int])`
    - `step()` moves one grid cell along current path or triggers replan if blocked

> *Copilot prompt:* "Implement `RobotAgent` that follows a path and replans on obstacle detection."  

---

## 7. Simulator (`simulator.py`)

- **Class**: `Simulator`
  - Holds `grid`, `agent`, `obstacle_mgr`, and visualization config
  - **Method**: `run()` main loop:
    1. `obstacle_mgr.update()`
    2. `agent.step()`
    3. Render grid, obstacles, robot, and path overlay
    4. Sleep for fixed time interval

> *Copilot prompt:* "Create `Simulator` that loops update, step, and renders using Matplotlib animation."  

---

## 8. Testing (`tests/`)

- **Write pytest tests** for each module:
  - `test_grid_map.py`: free cells, neighbors
  - `test_search_algorithms.py`: known small grids
  - `test_path_planner.py`: path correctness for both algorithms

> *Copilot prompt:* "Generate pytest tests for `GridMap` and `search_algorithms` modules."  

---

## 9. Demo Script (`examples/demo.py`)

- Instantiate a `GridMap` with sample obstacles
- Create `Simulator` with agent and obstacles
- Run for a fixed number of steps and save animation as GIF

> *Copilot prompt:* "Write a demo script to initialize and run the simulation, saving output as GIF."  

---

## 10. README.md

- **Sections**:
  - Project description
  - Installation
  - Usage examples
  - Performance metrics
  - Extension ideas

> *Copilot prompt:* "Generate a README with overview, install, usage, and architecture diagram."  

---

## Usage

1. Place this file at the root of your repo.
2. In VSCode, open any module and invoke Copilot to generate each section with the prompts above.
3. Commit incrementally, test after each major step.
4. Customize simulation parameters as needed.

---

> **Tip**: Prefix your Copilot triggers with comments (`# `) so that suggestions align with your instructions.  
