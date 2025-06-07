"""
Enhanced grid map with support for different movement models.
Supports 4-connected, 8-connected, and custom movement patterns.
"""

from typing import Set, Tuple, List, Dict
import math
from .grid_map import GridMap
import enum


class MovementType(enum.Enum):
    FOUR_CONNECTED = "4-connected"
    EIGHT_CONNECTED = "8-connected"
    KNIGHT = "knight"
    # Add other movement types if needed


class EnhancedGridMap(GridMap):
    """
    Enhanced grid map supporting different movement models and cost functions.
    
    Attributes:
        width (int): Grid width
        height (int): Grid height
        static_obstacles (Set[Tuple[int, int]]): Set of obstacle coordinates
        movement_model (str): Movement model ('4-connected', '8-connected', 'custom')
        custom_moves (List[Tuple[int, int]]): Custom movement directions
        cost_map (Dict[Tuple[int, int], float]): Custom cost for each cell
    """
    
    def __init__(self, width: int, height: int, 
                 static_obstacles: Set[Tuple[int, int]] = None,
                 movement_model: str = '4-connected',
                 custom_moves: List[Tuple[int, int]] = None):
        """
        Initialize EnhancedGridMap with movement model.
        
        Args:
            width: Grid width
            height: Grid height
            static_obstacles: Set of (x, y) coordinates representing obstacles
            movement_model: Movement model ('4-connected', '8-connected', 'custom')
            custom_moves: Custom movement directions for 'custom' model
        """
        super().__init__(width, height, static_obstacles)
        self.movement_model = movement_model
        self.custom_moves = custom_moves or []
        self.cost_map: Dict[Tuple[int, int], float] = {}
        
        # Validate movement model
        if movement_model not in ['4-connected', '8-connected', 'custom']:
            raise ValueError("movement_model must be '4-connected', '8-connected', or 'custom'")
        
        if movement_model == 'custom' and not custom_moves:
            raise ValueError("custom_moves must be provided for 'custom' movement model")
    
    def neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get valid neighboring cells based on movement model.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            List[Tuple[int, int]]: List of valid neighbor coordinates
        """
        potential_neighbors = []
        
        if self.movement_model == '4-connected':
            # Standard 4-connected movement: up, down, left, right
            potential_neighbors = [
                (x, y - 1),  # up
                (x, y + 1),  # down
                (x - 1, y),  # left
                (x + 1, y)   # right
            ]
        
        elif self.movement_model == '8-connected':
            # 8-connected movement: includes diagonals
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:  # Skip current position
                        potential_neighbors.append((x + dx, y + dy))
        
        elif self.movement_model == 'custom':
            # Custom movement patterns
            for dx, dy in self.custom_moves:
                potential_neighbors.append((x + dx, y + dy))
        
        # Filter to only free cells
        return [(nx, ny) for nx, ny in potential_neighbors if self.is_free(nx, ny)]
    
    def neighbors_with_cost(self, x: int, y: int) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get valid neighboring cells with movement costs.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            List[Tuple[Tuple[int, int], float]]: List of (neighbor, cost) pairs
        """
        neighbors_list = []
        
        if self.movement_model == '4-connected':
            # Standard 4-connected movement with unit cost
            moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if self.is_free(nx, ny):
                    cost = self.get_movement_cost((x, y), (nx, ny))
                    neighbors_list.append(((nx, ny), cost))
        
        elif self.movement_model == '8-connected':
            # 8-connected movement with diagonal costs
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:  # Skip current position
                        nx, ny = x + dx, y + dy
                        if self.is_free(nx, ny):
                            cost = self.get_movement_cost((x, y), (nx, ny))
                            neighbors_list.append(((nx, ny), cost))
        
        elif self.movement_model == 'custom':
            # Custom movement patterns with specified costs
            for dx, dy in self.custom_moves:
                nx, ny = x + dx, y + dy
                if self.is_free(nx, ny):
                    cost = self.get_movement_cost((x, y), (nx, ny))
                    neighbors_list.append(((nx, ny), cost))
        
        return neighbors_list
    
    def get_movement_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """
        Calculate the cost of moving from one position to another.
        
        Args:
            from_pos: Starting position (x, y)
            to_pos: Destination position (x, y)
            
        Returns:
            float: Movement cost
        """
        # Check if custom cost is defined for destination
        if to_pos in self.cost_map:
            base_cost = self.cost_map[to_pos]
        else:
            base_cost = 1.0  # Default cost
        
        # Calculate distance-based cost multiplier
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])
        
        if dx == 0 and dy == 0:
            distance_multiplier = 0.0  # No movement
        elif dx + dy == 1:
            distance_multiplier = 1.0  # Orthogonal movement
        elif dx == 1 and dy == 1:
            distance_multiplier = math.sqrt(2)  # Diagonal movement
        else:
            # Custom movement - use Euclidean distance
            distance_multiplier = math.sqrt(dx**2 + dy**2)
        
        return base_cost * distance_multiplier
    
    def set_cell_cost(self, x: int, y: int, cost: float):
        """
        Set custom cost for a specific cell.
        
        Args:
            x: X coordinate
            y: Y coordinate
            cost: Movement cost for this cell
        """
        self.cost_map[(x, y)] = cost
    
    def set_terrain_costs(self, terrain_map: Dict[Tuple[int, int], float]):
        """
        Set terrain costs for multiple cells.
        
        Args:
            terrain_map: Dictionary mapping (x, y) to cost values
        """
        self.cost_map.update(terrain_map)
    
    def get_cell_cost(self, x: int, y: int) -> float:
        """Get the cost of a specific cell."""
        return self.cost_map.get((x, y), 1.0)
    
    def create_terrain_pattern(self, pattern_type: str, **kwargs):
        """
        Create common terrain patterns with different costs.
        
        Args:
            pattern_type: Type of pattern ('mud', 'hills', 'water', 'roads')
            **kwargs: Pattern-specific parameters
        """
        if pattern_type == 'mud':
            # Create muddy areas with higher movement cost
            mud_cost = kwargs.get('cost', 2.0)
            density = kwargs.get('density', 0.1)
            
            import random
            for x in range(self.width):
                for y in range(self.height):
                    if self.is_free(x, y) and random.random() < density:
                        self.set_cell_cost(x, y, mud_cost)
        
        elif pattern_type == 'hills':
            # Create hilly terrain with gradient costs
            center_x = kwargs.get('center_x', self.width // 2)
            center_y = kwargs.get('center_y', self.height // 2)
            max_cost = kwargs.get('max_cost', 3.0)
            radius = kwargs.get('radius', min(self.width, self.height) // 4)
            
            for x in range(self.width):
                for y in range(self.height):
                    if self.is_free(x, y):
                        distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if distance <= radius:
                            # Cost increases towards center (hill peak)
                            cost = 1.0 + (max_cost - 1.0) * (1.0 - distance / radius)
                            self.set_cell_cost(x, y, cost)
        
        elif pattern_type == 'water':
            # Create water areas with very high cost (almost impassable)
            water_cost = kwargs.get('cost', 10.0)
            regions = kwargs.get('regions', [])
            
            for region in regions:
                x_start, y_start, width, height = region
                for x in range(x_start, min(x_start + width, self.width)):
                    for y in range(y_start, min(y_start + height, self.height)):
                        if self.is_free(x, y):
                            self.set_cell_cost(x, y, water_cost)
        
        elif pattern_type == 'roads':
            # Create roads with lower movement cost
            road_cost = kwargs.get('cost', 0.5)
            roads = kwargs.get('roads', [])
            
            for road in roads:
                start, end = road
                # Simple line drawing for road
                x1, y1 = start
                x2, y2 = end
                
                # Bresenham's line algorithm (simplified)
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                x, y = x1, y1
                
                x_inc = 1 if x1 < x2 else -1
                y_inc = 1 if y1 < y2 else -1
                
                error = dx - dy
                
                while True:
                    if self.is_free(x, y):
                        self.set_cell_cost(x, y, road_cost)
                    
                    if x == x2 and y == y2:
                        break
                    
                    e2 = 2 * error
                    if e2 > -dy:
                        error -= dy
                        x += x_inc
                    if e2 < dx:
                        error += dx
                        y += y_inc


def create_custom_movement_pattern(pattern_name: str) -> List[Tuple[int, int]]:
    """
    Create predefined custom movement patterns.
    
    Args:
        pattern_name: Name of movement pattern
        
    Returns:
        List of (dx, dy) movement offsets
    """
    if pattern_name == 'king':
        # Chess king movement (8-connected)
        return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    elif pattern_name == 'knight':
        # Chess knight movement
        return [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    
    elif pattern_name == 'plus':
        # Plus-shaped movement (4-connected)
        return [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    elif pattern_name == 'cross':
        # Diagonal-only movement
        return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    elif pattern_name == 'extended':
        # Extended movement (up to 2 cells away)
        moves = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx != 0 or dy != 0:  # Skip center
                    moves.append((dx, dy))
        return moves
    
    elif pattern_name == 'hex':
        # Hexagonal grid movement (6-connected)
        return [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    
    else:
        raise ValueError(f"Unknown movement pattern: {pattern_name}")


# Example usage and factory function
def create_enhanced_grid(width: int, height: int, 
                        movement_type: str = '4-connected',
                        obstacles: Set[Tuple[int, int]] = None,
                        terrain_config: Dict = None) -> EnhancedGridMap:
    """
    Factory function to create enhanced grid maps with common configurations.
    
    Args:
        width: Grid width
        height: Grid height
        movement_type: Movement model type
        obstacles: Static obstacles
        terrain_config: Terrain configuration parameters
        
    Returns:
        Configured EnhancedGridMap instance
    """
    # Handle predefined movement patterns
    if movement_type in ['king', 'knight', 'plus', 'cross', 'extended', 'hex']:
        custom_moves = create_custom_movement_pattern(movement_type)
        grid = EnhancedGridMap(width, height, obstacles, 'custom', custom_moves)
    else:
        grid = EnhancedGridMap(width, height, obstacles, movement_type)
    
    # Apply terrain configuration if provided
    if terrain_config:
        for terrain_type, config in terrain_config.items():
            grid.create_terrain_pattern(terrain_type, **config)
    
    return grid
