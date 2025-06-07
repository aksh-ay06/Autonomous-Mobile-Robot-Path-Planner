"""
Advanced path planning algorithms for autonomous mobile robot navigation.
Implements RRT (Rapidly-exploring Random Trees) and other sampling-based algorithms.
"""

import random
import math
from typing import Tuple, List, Optional, Set
import numpy as np
from .grid_map import GridMap


class Node:
    """Node for tree-based algorithms like RRT."""
    
    def __init__(self, position: Tuple[float, float], parent: Optional['Node'] = None):
        self.position = position
        self.parent = parent
        self.children: List['Node'] = []
        self.cost = 0.0
        
    def add_child(self, child: 'Node'):
        """Add a child node."""
        child.parent = self
        self.children.append(child)


def euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def line_collision_check(start: Tuple[float, float], end: Tuple[float, float], 
                        grid: GridMap, resolution: float = 0.1) -> bool:
    """
    Check if a line segment collides with obstacles using discrete sampling.
    
    Args:
        start: Start position
        end: End position
        grid: GridMap instance
        resolution: Sampling resolution for collision checking
        
    Returns:
        bool: True if collision detected, False otherwise
    """
    distance = euclidean_distance(start, end)
    if distance == 0:
        return not grid.is_free(int(start[0]), int(start[1]))
    
    steps = int(distance / resolution) + 1
    
    for i in range(steps + 1):
        t = i / steps if steps > 0 else 0
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        
        # Check if this point is in collision
        grid_x, grid_y = int(round(x)), int(round(y))
        if not grid.is_free(grid_x, grid_y):
            return True
    
    return False


def rrt(start: Tuple[int, int], goal: Tuple[int, int], grid: GridMap,
        max_iterations: int = 2000, step_size: float = 1.0, 
        goal_bias: float = 0.1) -> List[Tuple[int, int]]:
    """
    RRT (Rapidly-exploring Random Tree) path planning algorithm.
    
    Args:
        start: Starting position (x, y)
        goal: Goal position (x, y)
        grid: GridMap instance
        max_iterations: Maximum number of iterations
        step_size: Maximum distance for new nodes
        goal_bias: Probability of sampling towards goal
        
    Returns:
        List[Tuple[int, int]]: Path from start to goal, empty if no path found
    """
    if not grid.is_free(start[0], start[1]) or not grid.is_free(goal[0], goal[1]):
        return []
    
    # Initialize tree with start node
    start_node = Node((float(start[0]), float(start[1])))
    nodes = [start_node]
    goal_pos = (float(goal[0]), float(goal[1]))
    
    for iteration in range(max_iterations):
        # Sample random point (with goal bias)
        if random.random() < goal_bias:
            sample = goal_pos
        else:
            sample = (
                random.uniform(0, grid.width - 1),
                random.uniform(0, grid.height - 1)
            )
        
        # Find nearest node in tree
        nearest_node = min(nodes, key=lambda n: euclidean_distance(n.position, sample))
        
        # Generate new node towards sample
        distance = euclidean_distance(nearest_node.position, sample)
        if distance <= step_size:
            new_pos = sample
        else:
            # Scale to step_size
            direction = (
                (sample[0] - nearest_node.position[0]) / distance,
                (sample[1] - nearest_node.position[1]) / distance
            )
            new_pos = (
                nearest_node.position[0] + direction[0] * step_size,
                nearest_node.position[1] + direction[1] * step_size
            )
        
        # Check if new position is valid and connection is collision-free
        grid_x, grid_y = int(round(new_pos[0])), int(round(new_pos[1]))
        if (grid.is_free(grid_x, grid_y) and 
            not line_collision_check(nearest_node.position, new_pos, grid)):
            
            new_node = Node(new_pos, nearest_node)
            nearest_node.add_child(new_node)
            nodes.append(new_node)
            
            # Check if we reached the goal
            if euclidean_distance(new_pos, goal_pos) <= step_size:
                # Try to connect directly to goal
                if not line_collision_check(new_pos, goal_pos, grid):
                    goal_node = Node(goal_pos, new_node)
                    new_node.add_child(goal_node)
                    
                    # Reconstruct path
                    path = []
                    current = goal_node
                    while current is not None:
                        path.append((int(round(current.position[0])), 
                                   int(round(current.position[1]))))
                        current = current.parent
                    path.reverse()
                    return path
    
    return []  # No path found


def rrt_star(start: Tuple[int, int], goal: Tuple[int, int], grid: GridMap,
             max_iterations: int = 2000, step_size: float = 1.0, 
             goal_bias: float = 0.1, search_radius: float = 2.0) -> List[Tuple[int, int]]:
    """
    RRT* (RRT Star) - optimized version of RRT that improves path quality.
    
    Args:
        start: Starting position (x, y)
        goal: Goal position (x, y)
        grid: GridMap instance
        max_iterations: Maximum number of iterations
        step_size: Maximum distance for new nodes
        goal_bias: Probability of sampling towards goal
        search_radius: Radius for rewiring nearby nodes
        
    Returns:
        List[Tuple[int, int]]: Path from start to goal, empty if no path found
    """
    if not grid.is_free(start[0], start[1]) or not grid.is_free(goal[0], goal[1]):
        return []
    
    # Initialize tree with start node
    start_node = Node((float(start[0]), float(start[1])))
    start_node.cost = 0.0
    nodes = [start_node]
    goal_pos = (float(goal[0]), float(goal[1]))
    goal_node = None
    
    for iteration in range(max_iterations):
        # Sample random point (with goal bias)
        if random.random() < goal_bias:
            sample = goal_pos
        else:
            sample = (
                random.uniform(0, grid.width - 1),
                random.uniform(0, grid.height - 1)
            )
        
        # Find nearest node in tree
        nearest_node = min(nodes, key=lambda n: euclidean_distance(n.position, sample))
        
        # Generate new node towards sample
        distance = euclidean_distance(nearest_node.position, sample)
        if distance <= step_size:
            new_pos = sample
        else:
            # Scale to step_size
            direction = (
                (sample[0] - nearest_node.position[0]) / distance,
                (sample[1] - nearest_node.position[1]) / distance
            )
            new_pos = (
                nearest_node.position[0] + direction[0] * step_size,
                nearest_node.position[1] + direction[1] * step_size
            )
        
        # Check if new position is valid and connection is collision-free
        grid_x, grid_y = int(round(new_pos[0])), int(round(new_pos[1]))
        if (grid.is_free(grid_x, grid_y) and 
            not line_collision_check(nearest_node.position, new_pos, grid)):
            
            # Find nearby nodes for potential rewiring
            nearby_nodes = [n for n in nodes 
                          if euclidean_distance(n.position, new_pos) <= search_radius]
            
            # Choose parent with minimum cost
            best_parent = nearest_node
            min_cost = nearest_node.cost + euclidean_distance(nearest_node.position, new_pos)
            
            for node in nearby_nodes:
                potential_cost = node.cost + euclidean_distance(node.position, new_pos)
                if (potential_cost < min_cost and 
                    not line_collision_check(node.position, new_pos, grid)):
                    best_parent = node
                    min_cost = potential_cost
            
            # Create new node
            new_node = Node(new_pos, best_parent)
            new_node.cost = min_cost
            best_parent.add_child(new_node)
            nodes.append(new_node)
            
            # Rewire nearby nodes through new node if it provides better cost
            for node in nearby_nodes:
                if node != best_parent:
                    potential_cost = new_node.cost + euclidean_distance(new_node.position, node.position)
                    if (potential_cost < node.cost and 
                        not line_collision_check(new_node.position, node.position, grid)):
                        # Rewire node through new_node
                        if node.parent:
                            node.parent.children.remove(node)
                        node.parent = new_node
                        node.cost = potential_cost
                        new_node.add_child(node)
            
            # Check if we can connect to goal
            if euclidean_distance(new_pos, goal_pos) <= step_size:
                if not line_collision_check(new_pos, goal_pos, grid):
                    goal_cost = new_node.cost + euclidean_distance(new_pos, goal_pos)
                    if goal_node is None or goal_cost < goal_node.cost:
                        if goal_node and goal_node.parent:
                            goal_node.parent.children.remove(goal_node)
                        goal_node = Node(goal_pos, new_node)
                        goal_node.cost = goal_cost
                        new_node.add_child(goal_node)
    
    # Reconstruct best path if goal was reached
    if goal_node is not None:
        path = []
        current = goal_node
        while current is not None:
            path.append((int(round(current.position[0])), 
                       int(round(current.position[1]))))
            current = current.parent
        path.reverse()
        return path
    
    return []  # No path found


def prm(start: Tuple[int, int], goal: Tuple[int, int], grid: GridMap,
        num_samples: int = 500, connection_radius: float = 2.0) -> List[Tuple[int, int]]:
    """
    PRM (Probabilistic Roadmap) path planning algorithm.
    
    Args:
        start: Starting position (x, y)
        goal: Goal position (x, y)
        grid: GridMap instance
        num_samples: Number of random samples to generate
        connection_radius: Radius for connecting nearby samples
        
    Returns:
        List[Tuple[int, int]]: Path from start to goal, empty if no path found
    """
    if not grid.is_free(start[0], start[1]) or not grid.is_free(goal[0], goal[1]):
        return []
    
    # Generate random samples
    samples = [(float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))]
    
    for _ in range(num_samples):
        x = random.uniform(0, grid.width - 1)
        y = random.uniform(0, grid.height - 1)
        
        if grid.is_free(int(round(x)), int(round(y))):
            samples.append((x, y))
    
    # Build roadmap by connecting nearby samples
    roadmap = {i: [] for i in range(len(samples))}
    
    for i, sample1 in enumerate(samples):
        for j, sample2 in enumerate(samples):
            if i != j and euclidean_distance(sample1, sample2) <= connection_radius:
                if not line_collision_check(sample1, sample2, grid):
                    roadmap[i].append(j)
                    roadmap[j].append(i)  # Undirected graph
    
    # Find path using Dijkstra on the roadmap
    import heapq
    
    start_idx = 0  # Start is first sample
    goal_idx = 1   # Goal is second sample
    
    # Dijkstra's algorithm on roadmap
    distances = {i: float('inf') for i in range(len(samples))}
    distances[start_idx] = 0.0
    came_from = {i: None for i in range(len(samples))}
    frontier = [(0.0, start_idx)]
    
    while frontier:
        current_dist, current_idx = heapq.heappop(frontier)
        
        if current_idx == goal_idx:
            break
        
        if current_dist > distances[current_idx]:
            continue
        
        for neighbor_idx in roadmap[current_idx]:
            distance = euclidean_distance(samples[current_idx], samples[neighbor_idx])
            new_dist = distances[current_idx] + distance
            
            if new_dist < distances[neighbor_idx]:
                distances[neighbor_idx] = new_dist
                came_from[neighbor_idx] = current_idx
                heapq.heappush(frontier, (new_dist, neighbor_idx))
    
    # Reconstruct path
    if distances[goal_idx] == float('inf'):
        return []  # No path found
    
    path_indices = []
    current_idx = goal_idx
    while current_idx is not None:
        path_indices.append(current_idx)
        current_idx = came_from[current_idx]
    path_indices.reverse()
    
    # Convert to grid coordinates
    path = [(int(round(samples[idx][0])), int(round(samples[idx][1]))) 
            for idx in path_indices]
    
    return path
