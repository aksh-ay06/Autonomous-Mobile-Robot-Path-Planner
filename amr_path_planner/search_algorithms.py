"""
Search algorithms for autonomous mobile robot path planning.
Implements Dijkstra and A* algorithms.
"""

import heapq
from typing import Tuple, List, Callable, Dict, Optional
from .grid_map import GridMap


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Manhattan distance between two points."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def dijkstra(start: Tuple[int, int], goal: Tuple[int, int], grid: GridMap) -> List[Tuple[int, int]]:
    """
    Dijkstra's algorithm for finding shortest path.
    
    Args:
        start: Starting position (x, y)
        goal: Goal position (x, y)
        grid: GridMap instance
        
    Returns:
        List[Tuple[int, int]]: Path from start to goal, empty if no path exists
    """
    if not grid.is_free(start[0], start[1]) or not grid.is_free(goal[0], goal[1]):
        return []
    
    # Priority queue: (cost, position)
    frontier = [(0, start)]
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    cost_so_far: Dict[Tuple[int, int], float] = {start: 0}
    
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        
        if current == goal:
            break
            
        for neighbor in grid.neighbors(current[0], current[1]):
            new_cost = cost_so_far[current] + 1  # Uniform cost of 1 per step
            
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor))
                came_from[neighbor] = current
    
    # Reconstruct path
    if goal not in came_from:
        return []  # No path found
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    
    return path


def astar(start: Tuple[int, int], goal: Tuple[int, int], grid: GridMap, 
          heuristic: Callable[[Tuple[int, int], Tuple[int, int]], float] = manhattan_distance) -> List[Tuple[int, int]]:
    """
    A* algorithm for finding shortest path with heuristic.
    
    Args:
        start: Starting position (x, y)
        goal: Goal position (x, y)
        grid: GridMap instance
        heuristic: Heuristic function (default: Manhattan distance)
        
    Returns:
        List[Tuple[int, int]]: Path from start to goal, empty if no path exists
    """
    if not grid.is_free(start[0], start[1]) or not grid.is_free(goal[0], goal[1]):
        return []
    
    # Priority queue: (f_score, position)
    frontier = [(0, start)]
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    g_score: Dict[Tuple[int, int], float] = {start: 0}
    f_score: Dict[Tuple[int, int], float] = {start: heuristic(start, goal)}
    
    while frontier:
        current_f, current = heapq.heappop(frontier)
        
        if current == goal:
            break
            
        for neighbor in grid.neighbors(current[0], current[1]):
            tentative_g_score = g_score[current] + 1  # Uniform cost of 1 per step
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(frontier, (f_score[neighbor], neighbor))
    
    # Reconstruct path
    if goal not in came_from:
        return []  # No path found
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    
    return path
