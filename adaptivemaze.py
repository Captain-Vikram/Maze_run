from mazegenerator import MazeGenerator
from playertracker import PlayerTracker
import json
import random

class AdaptiveMazeGame:
    def __init__(self, player_id):
        self.player_id = player_id
        self.current_level = 1
        self.player_skill = "beginner"
        self.performance_history = []
        self.maze_params = self._get_maze_parameters("beginner")
    
    def _get_maze_parameters(self, skill_level):
        """Dynamically adjust maze parameters based on skill and level, with a max size of 31x31."""
        base_sizes = {
            "beginner": 11,
            "intermediate": 15,
            "advanced": 21
        }
        algorithms = {
            "beginner": "dfs",
            "intermediate": "kruskal",
            "advanced": "wilson"
        }
        
        # Calculate parameters FIRST
        base_size = base_sizes.get(skill_level, 11)
        size_increase = 2 * (self.current_level - 1)
        new_size = base_size + size_increase
        
        # Clamp the size to a maximum of 31
        new_size = min(new_size, 31)
        
        # Track size changes
        self.prev_size = getattr(self, 'prev_size', 0)
        self.maze_shape_changed = (new_size != self.prev_size)
        self.prev_size = new_size
        
        return {
            "width": new_size,
            "height": new_size,
            "algorithm": algorithms[skill_level]
        }
    
    def generate_maze(self):
        """Generate a maze using the specified algorithm with proper start and exit placement."""
        valid_algorithms = ["dfs", "kruskal", "wilson"]
        algorithm = self.maze_params.get("algorithm", "dfs")
        if algorithm not in valid_algorithms:
            algorithm = "dfs"

        maze_gen = MazeGenerator(
            self.maze_params["width"],
            self.maze_params["height"],
            algorithm
        )
        self.maze = maze_gen.generate_maze()

        # Set entry point at the center
        center_x, center_y = self.maze_params["height"] // 2, self.maze_params["width"] // 2
        self.maze[center_x, center_y] = 2  # Mark as start

        # Find a valid exit position on the border (not corners & inside the grid)
        possible_exits = []

        # Check top and bottom row (excluding corners)
        for col in range(1, self.maze_params["width"] - 1):
            if self.maze[1, col] == 0:  # Ensure exit is inside, not on the boundary
                possible_exits.append((1, col))
            if self.maze[self.maze_params["height"] - 2, col] == 0:
                possible_exits.append((self.maze_params["height"] - 2, col))

        # Check left and right columns (excluding corners)
        for row in range(1, self.maze_params["height"] - 1):
            if self.maze[row, 1] == 0:  # Ensure exit is inside, not on the boundary
                possible_exits.append((row, 1))
            if self.maze[row, self.maze_params["width"] - 2] == 0:
                possible_exits.append((row, self.maze_params["width"] - 2))

        # Ensure the exit has at least 2 open paths
        valid_exits = []
        for ex, ey in possible_exits:
            open_paths = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check 4 directions
                nx, ny = ex + dx, ey + dy
                if 0 <= nx < self.maze_params["height"] and 0 <= ny < self.maze_params["width"]:
                    if self.maze[nx, ny] == 0:
                        open_paths += 1

            if open_paths >= 2:
                valid_exits.append((ex, ey))

        # If valid exits exist, randomly choose one
        if valid_exits:
            exit_x, exit_y = random.choice(valid_exits)
            self.maze[exit_x, exit_y] = 3  # Mark as exit

        return self.maze, maze_gen

    def create_player_tracker(self):
        """Create a player tracker for the current maze."""
        return PlayerTracker(
            self.player_id,
            self.maze_params["width"],
            self.maze_params["height"]
        )
    
    def update_difficulty(self, performance_data):
        """Adjust difficulty based on performance metrics."""
        self.performance_history.append(performance_data)
        
        # Update skill level
        completion_time = performance_data.get("completion_time", 120)
        if completion_time < 60:
            self.player_skill = "advanced"
        elif completion_time < 120:
            self.player_skill = "intermediate"
        else:
            self.player_skill = "beginner"

        # Progress to next level and update maze parameters
        self.current_level += 1
        self.maze_params = self._get_maze_parameters(self.player_skill)
    
    def get_game_stats(self):
        """Get game statistics."""
        return {
            "player_id": self.player_id,
            "current_level": self.current_level,
            "player_skill": self.player_skill,
            "performance_history": self.performance_history
        }
    
    def save_game_stats(self, filename="game_stats.json"):
        """Save game statistics to a file."""
        with open(filename, 'w') as f:
            json.dump(self.get_game_stats(), f, indent=2)

    @property
    def difficulty(self):
        return self.player_skill
