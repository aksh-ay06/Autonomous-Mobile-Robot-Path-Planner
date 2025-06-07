"""
GridMap module for autonomous mobile robot path planning.
Provides a 2D grid representation with obstacle handling.
"""

from typing import Set, Tuple, List


class GridMap:
    """
    A 2D grid map for robot navigation with static obstacles.
    
    Attributes:
        width (int): Grid width
        height (int): Grid height
        static_obstacles (Set[Tuple[int, int]]): Set of obstacle coordinates
    """
    
    def __init__(self, width: int, height: int, static_obstacles: Set[Tuple[int, int]] = None):
        """
        Initialize GridMap with dimensions and optional obstacles.
        
        Args:
            width: Grid width
            height: Grid height  
            static_obstacles: Set of (x, y) coordinates representing obstacles
        """
        self.width = width
        self.height = height
        self.static_obstacles = static_obstacles or set()
    
    def is_free(self, x: int, y: int) -> bool:
        """
        Check if a grid cell is free (not an obstacle and within bounds).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            bool: True if cell is free, False otherwise
        """
        # Check bounds
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        # Check if obstacle
        return (x, y) not in self.static_obstacles
    
    def neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get valid neighboring cells (4-connected grid).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            List[Tuple[int, int]]: List of valid neighbor coordinates
        """
        # 4-connected neighbors: up, down, left, right
        potential_neighbors = [
            (x, y - 1),  # up
            (x, y + 1),  # down
            (x - 1, y),  # left
            (x + 1, y)   # right
        ]
        
        # Filter to only free cells
        return [(nx, ny) for nx, ny in potential_neighbors if self.is_free(nx, ny)]
    
    def add_obstacle(self, x: int, y: int):
        """Add an obstacle to the grid."""
        self.static_obstacles.add((x, y))
    
    def remove_obstacle(self, x: int, y: int):
        """Remove an obstacle from the grid."""
        self.static_obstacles.discard((x, y))
