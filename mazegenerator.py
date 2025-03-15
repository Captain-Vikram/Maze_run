import random
import numpy as np
import json
from collections import deque

class MazeGenerator:
    def __init__(self, width, height, algorithm='dfs'):
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        self.algorithm = algorithm
        self.maze = np.ones((self.height, self.width), dtype=int)
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Left, Down, Right, Up
    
    def generate_maze(self):
        """Generate a fully connected and playable maze"""
        # Reset maze to all walls
        self.maze = np.ones((self.height, self.width), dtype=int)
        
        # Choose algorithm and generate base maze
        if self.algorithm == 'dfs':
            self._depth_first_search()
        elif self.algorithm == 'kruskal':
            self._kruskal_algorithm()
        elif self.algorithm == 'wilson':
            self._wilson_algorithm()
        else:
            raise ValueError("Algorithm must be 'dfs', 'kruskal', or 'wilson'")
        
        # Add entry and exit points
        self._create_entry_exit()
        
        # Validate maze is playable
        if not self._validate_maze():
            # If not playable, retry with a different seed
            return self.generate_maze()
            
        return self.maze
    
    def _depth_first_search(self):
        """Generate maze using depth-first search with a growing tree algorithm"""
        # Create a grid of cells (each cell is a 2x2 area in the maze)
        cell_height = (self.height - 1) // 2
        cell_width = (self.width - 1) // 2
        
        # Start with random cell and mark it as visited
        start_y = random.randrange(cell_height)
        start_x = random.randrange(cell_width)
        visited = np.zeros((cell_height, cell_width), dtype=bool)
        visited[start_y, start_x] = True
        
        # Clear the path at the starting cell
        self.maze[start_y*2+1, start_x*2+1] = 0
        
        # Stack for backtracking
        stack = [(start_y, start_x)]
        
        while stack:
            y, x = stack[-1]
            
            # Get unvisited neighbors
            neighbors = []
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < cell_height and 0 <= nx < cell_width and not visited[ny, nx]:
                    neighbors.append((ny, nx, dy, dx))
            
            if neighbors:
                # Choose a random unvisited neighbor
                ny, nx, dy, dx = random.choice(neighbors)
                
                # Remove the wall between current cell and chosen neighbor
                self.maze[y*2+1+dy, x*2+1+dx] = 0
                
                # Mark the new cell as part of the path
                self.maze[ny*2+1, nx*2+1] = 0
                
                # Mark as visited and push to stack
                visited[ny, nx] = True
                stack.append((ny, nx))
            else:
                # Dead end, backtrack
                stack.pop()
    
    def _kruskal_algorithm(self):
        """Generate maze using randomized Kruskal's algorithm"""
        # Create a grid where each cell is a node in the MST
        cell_height = (self.height - 1) // 2
        cell_width = (self.width - 1) // 2
        
        # Initialize disjoint set for tracking connected components
        parent = {}
        rank = {}
        
        # Initialize each cell as its own set
        for y in range(cell_height):
            for x in range(cell_width):
                parent[(y, x)] = (y, x)
                rank[(y, x)] = 0
                # Clear the path at each cell
                self.maze[y*2+1, x*2+1] = 0
        
        def find(node):
            # Path compression find
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]
        
        def union(node1, node2):
            # Union by rank
            root1 = find(node1)
            root2 = find(node2)
            
            if root1 != root2:
                if rank[root1] < rank[root2]:
                    parent[root1] = root2
                else:
                    parent[root2] = root1
                    if rank[root1] == rank[root2]:
                        rank[root1] += 1
                return True
            return False
        
        # Create list of all possible walls
        walls = []
        for y in range(cell_height):
            for x in range(cell_width):
                if x < cell_width - 1:
                    walls.append((y, x, y, x+1))  # Horizontal wall
                if y < cell_height - 1:
                    walls.append((y, x, y+1, x))  # Vertical wall
        
        # Shuffle walls for randomization
        random.shuffle(walls)
        
        # Process each wall
        for y1, x1, y2, x2 in walls:
            if union((y1, x1), (y2, x2)):
                # Remove the wall between cells
                wall_y = y1*2+1 + (y2-y1)
                wall_x = x1*2+1 + (x2-x1)
                self.maze[wall_y, wall_x] = 0
    
    def _wilson_algorithm(self):
        """Generate maze using Wilson's algorithm (loop-erased random walks)"""
        # Create a grid of cells
        cell_height = (self.height - 1) // 2
        cell_width = (self.width - 1) // 2
        
        # Start with all cells unvisited
        visited = np.zeros((cell_height, cell_width), dtype=bool)
        
        # Choose random starting cell and mark as visited
        y = random.randrange(cell_height)
        x = random.randrange(cell_width)
        visited[y, x] = True
        
        # Clear the path at the starting cell
        self.maze[y*2+1, x*2+1] = 0
        
        # List of unvisited cells
        unvisited = [(i, j) for i in range(cell_height) for j in range(cell_width) if not visited[i, j]]
        
        while unvisited:
            # Pick a random unvisited cell to start walk
            y, x = random.choice(unvisited)
            path = [(y, x)]
            
            # Perform a random walk until hitting a visited cell
            while not visited[y, x]:
                dy, dx = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
                ny, nx = y + dy, x + dx
                
                # If out of bounds, pick another direction
                if not (0 <= ny < cell_height and 0 <= nx < cell_width):
                    continue
                
                # Check if this position creates a loop in our path
                if (ny, nx) in path:
                    # Erase the loop by removing all steps after this position
                    loop_index = path.index((ny, nx))
                    path = path[:loop_index+1]
                else:
                    path.append((ny, nx))
                
                y, x = ny, nx
            
            # Add the path to the maze
            for i in range(len(path)-1):
                cy, cx = path[i]
                ny, nx = path[i+1]
                
                # Clear the cells
                self.maze[cy*2+1, cx*2+1] = 0
                self.maze[ny*2+1, nx*2+1] = 0
                
                # Clear the wall between them
                wall_y = cy*2+1 + (ny-cy)
                wall_x = cx*2+1 + (nx-cx)
                self.maze[wall_y, wall_x] = 0
                
                # Mark as visited
                visited[cy, cx] = True
            
            # Update unvisited list
            unvisited = [(i, j) for i in range(cell_height) for j in range(cell_width) if not visited[i, j]]
    
    def _create_entry_exit(self):
        """Create entry and exit points with guaranteed connectivity"""
        # First ensure we have a fully connected maze
        # Find all path cells (value 0)
        path_cells = [(y, x) for y in range(1, self.height-1) 
                    for x in range(1, self.width-1) if self.maze[y, x] == 0]
        
        if not path_cells:
            # If no path cells, create a simple path from top-left to bottom-right
            for y in range(1, self.height-1, 2):
                for x in range(1, self.width-1):
                    self.maze[y, x] = 0
            for x in range(1, self.width-1, 2):
                for y in range(1, self.height-1):
                    self.maze[y, x] = 0
            path_cells = [(y, x) for y in range(1, self.height-1) 
                        for x in range(1, self.width-1) if self.maze[y, x] == 0]
        
        # Avoid central area to prevent the problematic pattern
        center_y, center_x = self.height // 2, self.width // 2
        buffer = max(2, min(self.height, self.width) // 10)  # Scale buffer with maze size
        
        # Filter out cells too close to center
        valid_cells = [(y, x) for y, x in path_cells if 
                    abs(y - center_y) > buffer or abs(x - center_x) > buffer]
        
        if not valid_cells:
            valid_cells = path_cells  # Fallback if all cells are near center
        
        # Find paths to edges
        edge_paths = []
        for y, x in valid_cells:
            # Check if this cell can connect to an edge
            for direction, (dy, dx) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
                can_connect = True
                cy, cx = y, x
                
                # Try to trace a path to an edge
                while 0 < cy < self.height-1 and 0 < cx < self.width-1:
                    cy += dy
                    cx += dx
                    if self.maze[cy][cx] == 1:  # Hit a wall
                        can_connect = False
                        break
                
                if can_connect:  # Found a path to edge
                    edge_y, edge_x = cy, cx
                    if (edge_y == 0 or edge_y == self.height-1 or 
                        edge_x == 0 or edge_x == self.width-1):
                        edge_paths.append((y, x, edge_y, edge_x, direction))
        
        # If no edge paths found, create some
        if not edge_paths:
            # Create at least one path from a valid cell to an edge
            if valid_cells:
                y, x = random.choice(valid_cells)
                
                # Pick a random direction and make a path to the edge
                direction = random.randint(0, 3)
                dy, dx = [(0, -1), (1, 0), (0, 1), (-1, 0)][direction]
                
                cy, cx = y, x
                while 0 < cy < self.height-1 and 0 < cx < self.width-1:
                    cy += dy
                    cx += dx
                    self.maze[cy][cx] = 0  # Make path
                
                edge_paths.append((y, x, cy, cx, direction))
        
        # Choose entry and exit from different sides if possible
        if len(edge_paths) >= 2:
            # Sort paths by the direction they lead to
            edge_paths.sort(key=lambda p: p[4])
            
            # Try to pick entry and exit from opposite sides
            entry_path = edge_paths[0]
            for path in reversed(edge_paths):
                if abs(path[4] - entry_path[4]) == 2:  # Opposite directions
                    exit_path = path
                    break
            else:
                # No opposite paths, just pick the last one
                exit_path = edge_paths[-1]
            
            self.entry_point = (entry_path[2], entry_path[3])  # Edge coordinates
            self.exit_point = (exit_path[2], exit_path[3])
        else:
            # Only one path, create entry and exit on opposite sides
            y, x, edge_y, edge_x, direction = edge_paths[0]
            
            # Entry at the found edge
            self.entry_point = (edge_y, edge_x)
            
            # Create exit on the opposite side
            opposite_dir = (direction + 2) % 4
            opp_dy, opp_dx = [(0, -1), (1, 0), (0, 1), (-1, 0)][opposite_dir]
            
            # Trace from center to opposite edge
            cy, cx = y, x
            while 0 < cy < self.height-1 and 0 < cx < self.width-1:
                cy += opp_dy 
                cx += opp_dx
                self.maze[cy][cx] = 0  # Make path
            
            self.exit_point = (cy, cx)
        
        # Ensure entry and exit points are open
        self.maze[self.entry_point] = 0
        self.maze[self.exit_point] = 0

        # === ADD THIS BLOCK HERE ===
        # Create outer border walls
        for y in range(self.height):
            for x in range(self.width):
                if y == 0 or y == self.height-1 or x == 0 or x == self.width-1:
                    if (y, x) not in [self.entry_point, self.exit_point]:
                        self.maze[y, x] = 1

    def _find_nearest_path(self, y, x):
        """Find the nearest path cell from given coordinates"""
        queue = deque([(y, x, 0)])  # y, x, distance
        visited = set([(y, x)])
        
        while queue:
            cy, cx, dist = queue.popleft()
            
            # If we found a path cell, return it
            if self.maze[cy, cx] == 0:
                return (cy, cx)
            
            # Check neighbors
            for dy, dx in self.directions:
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and 
                    (ny, nx) not in visited):
                    visited.add((ny, nx))
                    queue.append((ny, nx, dist + 1))
        
        # If no path cell found nearby
        return None
    
    def _validate_maze(self):
        """Check if the maze is fully connected and playable"""
        if not hasattr(self, 'entry_point') or not hasattr(self, 'exit_point'):
            return False
        
        # Check if entry and exit are accessible
        if (self.maze[self.entry_point] != 0 or 
            self.maze[self.exit_point] != 0):
            return False
        
        # BFS to ensure there's a path from entry to exit
        queue = deque([self.entry_point])
        visited = set([self.entry_point])
        
        while queue:
            y, x = queue.popleft()
            
            # Check if we've reached the exit
            if (y, x) == self.exit_point:
                return True
            
            # Check all four directions
            for dy, dx in self.directions:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and 
                    self.maze[ny, nx] == 0 and (ny, nx) not in visited):
                    visited.add((ny, nx))
                    queue.append((ny, nx))
        
        # If we can't reach the exit from the entry
        return False
    
    def _check_maze_complexity(self):
        """Ensure the maze has adequate complexity"""
        # Count number of junctions (cells with 3+ open neighbors)
        junctions = 0
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if self.maze[y, x] == 0:
                    open_neighbors = sum(1 for dy, dx in self.directions 
                                     if self.maze[y+dy, x+dx] == 0)
                    if open_neighbors >= 3:
                        junctions += 1
        
        # Calculate maze size and expected minimum junctions
        maze_size = self.width * self.height
        min_junctions = max(1, maze_size // 100)
        
        return junctions >= min_junctions
    
    def add_player_position(self, position=None):
        """Add a player position to the maze, ensuring it's on a valid path"""
        if position:
            y, x = position
            if 0 <= y < self.height and 0 <= x < self.width and self.maze[y, x] == 0:
                self.player_position = (y, x)
                return
        
        # If no valid position provided, place near entry but not at entry
        visited = set([self.entry_point])
        queue = deque([self.entry_point])
        
        while queue:
            y, x = queue.popleft()
            
            # Skip the entry point itself for better gameplay
            if (y, x) != self.entry_point:
                # Place player a few steps from entry
                self.player_position = (y, x)
                return
            
            # Try neighbors
            for dy, dx in self.directions:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and 
                    self.maze[ny, nx] == 0 and (ny, nx) not in visited):
                    visited.add((ny, nx))
                    queue.append((ny, nx))
        
        # Fallback to entry point if no other valid position found
        self.player_position = self.entry_point
    
    def save_maze_to_json(self, filename="maze.json"):
        """Save the generated maze as a JSON file."""
        maze_dict = {
            "width": self.width,
            "height": self.height,
            "maze": self.maze.tolist(),
            "entry": list(self.entry_point) if hasattr(self, 'entry_point') else None,
            "exit": list(self.exit_point) if hasattr(self, 'exit_point') else None,
            "player": list(self.player_position) if hasattr(self, 'player_position') else None
        }
        with open(filename, 'w') as f:
            json.dump(maze_dict, f, indent=2)
    
    def display_maze(self, show_player=True):
        """Display the maze in console with ASCII characters"""
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                if show_player and hasattr(self, 'player_position') and (y, x) == self.player_position:
                    line += "P "  # Player
                elif (y, x) == self.entry_point:
                    line += "S "  # Start
                elif (y, x) == self.exit_point:
                    line += "E "  # Exit
                elif self.maze[y, x] == 1:
                    line += "██"  # Wall
                else:
                    line += "  "  # Path
            print(line)