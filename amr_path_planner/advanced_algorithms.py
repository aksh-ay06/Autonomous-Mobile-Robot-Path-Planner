"""
Advanced path planning algorithms for autonomous mobile robot navigation.
Includes sampling-based planners: RRT, RRT*, PRM.

Assumptions:
- GridMap uses integer (x, y) cells.
- Collision checking uses grid line traversal between integer cells.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeAlias

from .grid_map import GridMap
from .geometry import bresenham_cells, euclidean_distance, line_collision_check

Point: TypeAlias = tuple[int, int]
FPoint: TypeAlias = tuple[float, float]


def _steer(from_pos: FPoint, to_pos: FPoint, step_size: float) -> FPoint:
    d = euclidean_distance(from_pos, to_pos)
    if d <= step_size:
        return to_pos
    if d == 0:
        return from_pos
    ux = (to_pos[0] - from_pos[0]) / d
    uy = (to_pos[1] - from_pos[1]) / d
    return (from_pos[0] + ux * step_size, from_pos[1] + uy * step_size)


def _sample_free_point(rng: random.Random, grid: GridMap) -> FPoint:
    # sample uniformly over grid extents (continuous)
    return (rng.uniform(0, grid.width - 1), rng.uniform(0, grid.height - 1))


# ----------------------------
# RRT / RRT*
# ----------------------------

@dataclass
class Node:
    pos: FPoint
    parent: Optional["Node"] = None
    children: list["Node"] = field(default_factory=list)
    cost: float = 0.0  # cost-to-come

    # Provide compatibility alias expected by tests
    @property
    def position(self) -> FPoint:
        return self.pos

    def attach_child(self, child: "Node") -> None:
        child.parent = self
        self.children.append(child)


def _nearest(nodes: Sequence[Node], sample: FPoint) -> Node:
    # O(n) nearest neighbor; fine for small maps; swap for KDTree later if needed.
    return min(nodes, key=lambda n: euclidean_distance(n.pos, sample))


def _propagate_costs(root: Node) -> None:
    """After rewiring, update costs of all descendants."""
    for child in root.children:
        child.cost = root.cost + euclidean_distance(root.pos, child.pos)
        _propagate_costs(child)


def rrt(
    start: Point,
    goal: Point,
    grid: GridMap,
    max_iterations: int = 2000,
    step_size: float = 1.0,
    goal_bias: float = 0.1,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> list[Point]:
    """
    RRT planner. Returns a grid path (int points) or [] if not found.
    """
    if not grid.is_free(*start) or not grid.is_free(*goal):
        return []

    rng = rng or random.Random(seed)
    start_node = Node((float(start[0]), float(start[1])), parent=None, cost=0.0)
    nodes: list[Node] = [start_node]
    goal_pos: FPoint = (float(goal[0]), float(goal[1]))

    for _ in range(max_iterations):
        sample = goal_pos if rng.random() < goal_bias else _sample_free_point(rng, grid)
        nearest = _nearest(nodes, sample)
        new_pos = _steer(nearest.pos, sample, step_size)

        gx, gy = int(round(new_pos[0])), int(round(new_pos[1]))
        if not grid.is_free(gx, gy):
            continue
        if line_collision_check(nearest.pos, new_pos, grid):
            continue

        new_node = Node(new_pos, parent=nearest, cost=nearest.cost + euclidean_distance(nearest.pos, new_pos))
        nearest.attach_child(new_node)
        nodes.append(new_node)

        # goal connection
        if euclidean_distance(new_pos, goal_pos) <= step_size and not line_collision_check(new_pos, goal_pos, grid):
            goal_node = Node(goal_pos, parent=new_node, cost=new_node.cost + euclidean_distance(new_pos, goal_pos))
            new_node.attach_child(goal_node)
            return _backtrack_path(goal_node)

    return []


def rrt_star(
    start: Point,
    goal: Point,
    grid: GridMap,
    max_iterations: int = 2000,
    step_size: float = 1.0,
    goal_bias: float = 0.1,
    search_radius: float = 2.0,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> list[Point]:
    """
    RRT* planner with rewiring and cost propagation.
    Returns best-found grid path (int points) or [] if not found.
    """
    if not grid.is_free(*start) or not grid.is_free(*goal):
        return []

    rng = rng or random.Random(seed)
    start_node = Node((float(start[0]), float(start[1])), parent=None, cost=0.0)
    nodes: list[Node] = [start_node]
    goal_pos: FPoint = (float(goal[0]), float(goal[1]))
    best_goal: Optional[Node] = None

    for _ in range(max_iterations):
        sample = goal_pos if rng.random() < goal_bias else _sample_free_point(rng, grid)
        nearest = _nearest(nodes, sample)
        new_pos = _steer(nearest.pos, sample, step_size)

        gx, gy = int(round(new_pos[0])), int(round(new_pos[1]))
        if not grid.is_free(gx, gy):
            continue
        if line_collision_check(nearest.pos, new_pos, grid):
            continue

        # find nearby nodes for parent selection + rewiring
        nearby = [n for n in nodes if euclidean_distance(n.pos, new_pos) <= search_radius]
        if not nearby:
            nearby = [nearest]

        # choose best parent by min cost + collision-free edge
        best_parent = nearest
        best_cost = nearest.cost + euclidean_distance(nearest.pos, new_pos)

        for n in nearby:
            candidate_cost = n.cost + euclidean_distance(n.pos, new_pos)
            if candidate_cost < best_cost and not line_collision_check(n.pos, new_pos, grid):
                best_parent = n
                best_cost = candidate_cost

        new_node = Node(new_pos, parent=best_parent, cost=best_cost)
        best_parent.attach_child(new_node)
        nodes.append(new_node)

        # rewire: if going through new_node improves nearby nodes, reattach them
        for n in nearby:
            if n is new_node or n is best_parent:
                continue
            new_cost = new_node.cost + euclidean_distance(new_node.pos, n.pos)
            if new_cost < n.cost and not line_collision_check(new_node.pos, n.pos, grid):
                # detach from old parent
                if n.parent is not None:
                    try:
                        n.parent.children.remove(n)
                    except ValueError:
                        pass
                # attach to new parent
                new_node.attach_child(n)
                n.cost = new_cost
                _propagate_costs(n)  # IMPORTANT: keep subtree costs consistent

        # attempt goal connection / keep best
        if euclidean_distance(new_pos, goal_pos) <= step_size and not line_collision_check(new_pos, goal_pos, grid):
            goal_cost = new_node.cost + euclidean_distance(new_pos, goal_pos)
            if best_goal is None or goal_cost < best_goal.cost:
                # replace best goal node
                best_goal = Node(goal_pos, parent=new_node, cost=goal_cost)

    return _backtrack_path(best_goal) if best_goal is not None else []


def _backtrack_path(node: Optional[Node]) -> list[Point]:
    if node is None:
        return []
    path: list[Point] = []
    cur: Optional[Node] = node
    while cur is not None:
        path.append((int(round(cur.pos[0])), int(round(cur.pos[1]))))
        cur = cur.parent
    path.reverse()
    return _dedupe_consecutive(path)


def _dedupe_consecutive(path: list[Point]) -> list[Point]:
    if not path:
        return []
    out = [path[0]]
    for p in path[1:]:
        if p != out[-1]:
            out.append(p)
    return out


# ----------------------------
# PRM
# ----------------------------

def prm(
    start: Point,
    goal: Point,
    grid: GridMap,
    num_samples: int = 500,
    connection_radius: float = 2.0,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> list[Point]:
    """
    PRM planner:
    - sample free points
    - connect edges within radius if collision-free
    - run Dijkstra over roadmap graph
    - return path in grid coordinates
    """
    if not grid.is_free(*start) or not grid.is_free(*goal):
        return []

    rng = rng or random.Random(seed)

    samples: list[FPoint] = [(float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))]

    # sample free cells
    for _ in range(num_samples):
        p = _sample_free_point(rng, grid)
        gx, gy = int(round(p[0])), int(round(p[1]))
        if grid.is_free(gx, gy):
            samples.append(p)

    n = len(samples)
    roadmap: dict[int, list[int]] = {i: [] for i in range(n)}

    # build undirected graph without duplicate edges
    for i in range(n):
        for j in range(i + 1, n):
            if euclidean_distance(samples[i], samples[j]) <= connection_radius:
                if not line_collision_check(samples[i], samples[j], grid):
                    roadmap[i].append(j)
                    roadmap[j].append(i)

    # Dijkstra on roadmap (weighted by Euclidean distance)
    start_idx, goal_idx = 0, 1
    dist: list[float] = [float("inf")] * n
    prev: list[Optional[int]] = [None] * n
    dist[start_idx] = 0.0

    pq: list[tuple[float, int]] = [(0.0, start_idx)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == goal_idx:
            break
        for v in roadmap[u]:
            w = euclidean_distance(samples[u], samples[v])
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dist[goal_idx] == float("inf"):
        return []

    # reconstruct indices
    idx_path: list[int] = []
    cur = goal_idx
    while cur is not None:
        idx_path.append(cur)
        cur = prev[cur]
    idx_path.reverse()

    rounded: list[Point] = [(int(round(samples[i][0])), int(round(samples[i][1]))) for i in idx_path]

    # Expand each edge with Bresenham cells to maintain collision-checked connectivity after rounding.
    grid_path: list[Point] = []
    for p in rounded:
        if not grid_path:
            grid_path.append(p)
            continue
        segment = bresenham_cells(grid_path[-1], p)
        if segment:
            grid_path.extend(segment[1:])  # skip duplicate start
        else:
            grid_path.append(p)

    return _dedupe_consecutive(grid_path)
