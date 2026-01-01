import argparse
import random
from typing import List, Set, Tuple

from amr_path_planner import (
    GridMap,
    PathPlanner,
    RobotAgent,
    MultiRobotCoordinator,
    Simulator,
    DynamicObstacleMgr,
)

Point = Tuple[int, int]


def sample_positions(rng: random.Random, width: int, height: int, count: int, min_sep: int = 5) -> List[Point]:
    """Sample distinct grid cells; enforce a loose separation so robots spread out."""
    chosen: List[Point] = []
    attempts = 0
    max_attempts = count * 100
    while len(chosen) < count and attempts < max_attempts:
        p = (rng.randrange(width), rng.randrange(height))
        if any(abs(p[0] - q[0]) + abs(p[1] - q[1]) < min_sep for q in chosen):
            attempts += 1
            continue
        chosen.append(p)
    # If we couldn't satisfy min_sep, fill remaining without the constraint.
    while len(chosen) < count:
        chosen.append((rng.randrange(width), rng.randrange(height)))
    return chosen


def build_obstacles(rng: random.Random, width: int, height: int, count: int, reserved: Set[Point]) -> Set[Point]:
    """Create a random obstacle set while keeping reserved cells free."""
    obstacles: Set[Point] = set()
    attempts = 0
    max_attempts = count * 50
    while len(obstacles) < count and attempts < max_attempts:
        p = (rng.randrange(width), rng.randrange(height))
        if p in reserved:
            attempts += 1
            continue
        obstacles.add(p)
    return obstacles


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-robot coordination demo")
    parser.add_argument("--width", type=int, default=20, help="Grid width")
    parser.add_argument("--height", type=int, default=20, help="Grid height")
    parser.add_argument("--robots", type=int, default=3, help="Number of robots")
    parser.add_argument("--obstacles", type=int, default=40, help="Number of static obstacles")
    parser.add_argument("--dynamic-obstacles", type=int, default=0, help="Number of dynamic obstacles (moving)")
    parser.add_argument("--dynamic-move-prob", type=float, default=0.7, help="Movement probability for dynamic obstacles")
    parser.add_argument("--horizon", type=int, default=50, help="Space-time planning horizon")
    parser.add_argument("--step-delay", type=float, default=0.1, help="Visualization delay (seconds)")
    parser.add_argument("--max-steps", type=int, default=300, help="Simulation step cap")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for reproducibility")
    parser.add_argument(
        "--mode",
        choices=["priority", "cooperative", "centralized"],
        default="priority",
        help="Coordination mode",
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Plan start/goal pairs
    starts = sample_positions(rng, args.width, args.height, args.robots)
    goals = sample_positions(rng, args.width, args.height, args.robots)

    reserved = set(starts + goals)
    obstacles = build_obstacles(rng, args.width, args.height, args.obstacles, reserved)

    grid = GridMap(args.width, args.height, static_obstacles=obstacles)

    coord = MultiRobotCoordinator(grid, coordination_mode=args.mode, horizon=args.horizon)

    obstacle_mgr = None
    if args.dynamic_obstacles > 0:
        # Prevent dynamic obstacles from spawning on start/goal cells via blocked_fn.
        def _blocked(x: int, y: int) -> bool:
            return (x, y) in reserved

        obstacle_mgr = DynamicObstacleMgr(
            grid=grid,
            movement_probability=args.dynamic_move_prob,
            blocked_fn=_blocked,
        )
        obstacle_mgr.spawn_random_obstacles(args.dynamic_obstacles)

    for start, goal in zip(starts, goals):
        planner = PathPlanner(grid=grid)
        robot = RobotAgent(position=start, planner=planner)
        rid = coord.add_robot(robot)
        coord.set_robot_goal(rid, goal)

    coord.plan_coordinated_paths(obstacle_mgr)

    sim = Simulator(
        grid=grid,
        multi_robot_coordinator=coord,
        obstacle_mgr=obstacle_mgr,
        max_steps=args.max_steps,
        step_delay=args.step_delay,
    )
    sim.run(visualize=True, save_gif=False)


if __name__ == "__main__":
    main()