from amr_path_planner import (
    GridMap,
    PathPlanner,
    RobotAgent,
    Simulator,
)

# --------------------
# Create grid
# --------------------
width, height = 20, 15
static_obstacles = {
    (5, y) for y in range(3, 12)
} | {
    (10, y) for y in range(0, 8)
}

grid = GridMap(width, height, static_obstacles)

# --------------------
# Planner + robot
# --------------------
planner = PathPlanner(
    algorithm="astar",
    grid=grid,
)

robot = RobotAgent(
    position=(1, 1),
    planner=planner,
)

robot.plan_to((18, 13))

# --------------------
# Simulator
# --------------------
sim = Simulator(
    grid=grid,
    agent=robot,
    step_delay=0.15,
    max_steps=500,
)

sim.run(visualize=True)
