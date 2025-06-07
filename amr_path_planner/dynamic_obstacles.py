"""
Dynamic obstacles manager for autonomous mobile robot path planning.
Handles moving obstacles with random walk behavior.
"""

import random
from typing import List, Tuple, Set
from .grid_map import GridMap


class DynamicObstacleMgr:
    """
    Manager for dynamic obstacles that move randomly in the grid.
    
    Attributes:
        obstacles (List[Tuple[int, int]]): Current positions of dynamic obstacles
        grid (GridMap): Reference to the grid map
        movement_probability (float): Probability of movement per update
    """
    
    def __init__(self, grid: GridMap, initial_obstacles: List[Tuple[int, int]] = None, 
                 movement_probability: float = 0.7):
        """
        Initialize DynamicObstacleMgr.
        
        Args:
            grid: GridMap instance
            initial_obstacles: Initial positions of dynamic obstacles
            movement_probability: Probability that an obstacle moves each update (0.0 - 1.0)
        """
        self.grid = grid
        self.obstacles = initial_obstacles or []
        self.movement_probability = movement_probability
        self.obstacle_set = set(self.obstacles)  # For fast collision checking
    
    def add_obstacle(self, x: int, y: int):
        """Add a dynamic obstacle at the specified position."""
        if self.grid.is_free(x, y) and (x, y) not in self.obstacle_set:
            self.obstacles.append((x, y))
            self.obstacle_set.add((x, y))
    
    def remove_obstacle(self, x: int, y: int):
        """Remove a dynamic obstacle from the specified position."""
        if (x, y) in self.obstacle_set:
            self.obstacles.remove((x, y))
            self.obstacle_set.remove((x, y))
    
    def update(self):
        """
        Update all dynamic obstacles by moving them randomly.
        Each obstacle has a chance to move to an adjacent free cell.
        """
        new_obstacles = []
        new_obstacle_set = set()
        
        for obs_x, obs_y in self.obstacles:
            # Decide if this obstacle should move
            if random.random() < self.movement_probability:
                # Get possible moves (including staying in place)
                possible_moves = [(obs_x, obs_y)]  # Can stay in current position
                
                # Add neighboring free cells that are not occupied by other obstacles
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_x, new_y = obs_x + dx, obs_y + dy
                    if (self.grid.is_free(new_x, new_y) and 
                        (new_x, new_y) not in self.grid.static_obstacles and
                        (new_x, new_y) not in new_obstacle_set):
                        possible_moves.append((new_x, new_y))
                
                # Choose random move
                if possible_moves:
                    new_pos = random.choice(possible_moves)
                    new_obstacles.append(new_pos)
                    new_obstacle_set.add(new_pos)
                else:
                    # If no valid moves, stay in place
                    new_obstacles.append((obs_x, obs_y))
                    new_obstacle_set.add((obs_x, obs_y))
            else:
                # Don't move this obstacle
                new_obstacles.append((obs_x, obs_y))
                new_obstacle_set.add((obs_x, obs_y))
        
        self.obstacles = new_obstacles
        self.obstacle_set = new_obstacle_set
    
    def is_collision(self, x: int, y: int) -> bool:
        """
        Check if the given position collides with any dynamic obstacle.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            bool: True if collision, False otherwise
        """
        return (x, y) in self.obstacle_set
    
    def get_obstacle_positions(self) -> List[Tuple[int, int]]:
        """Get current positions of all dynamic obstacles."""
        return self.obstacles.copy()
    
    def clear_obstacles(self):
        """Remove all dynamic obstacles."""
        self.obstacles.clear()
        self.obstacle_set.clear()
    
    def spawn_random_obstacles(self, count: int):
        """
        Spawn random obstacles in free spaces of the grid.
        
        Args:
            count: Number of obstacles to spawn
        """
        attempts = 0
        spawned = 0
        max_attempts = count * 10  # Prevent infinite loop
        
        while spawned < count and attempts < max_attempts:
            x = random.randint(0, self.grid.width - 1)
            y = random.randint(0, self.grid.height - 1)
            
            if (self.grid.is_free(x, y) and 
                (x, y) not in self.obstacle_set and
                (x, y) not in self.grid.static_obstacles):
                self.add_obstacle(x, y)
                spawned += 1
            
            attempts += 1
