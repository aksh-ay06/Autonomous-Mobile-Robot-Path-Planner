print("Starting test...")

try:
    from amr_path_planner.advanced_algorithms import Node, rrt, rrt_star, prm
    from amr_path_planner.grid_map import GridMap
    print("Imports successful")
    
    # Test Node
    node = Node((5, 5))
    print(f"Node created: {node.position}")
    
    # Test simple RRT
    grid = GridMap(10, 10)
    path = rrt((1, 1), (8, 8), grid, 100)
    print(f"RRT path found: {len(path) > 0 if path else False}")
    
    print("All tests completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
