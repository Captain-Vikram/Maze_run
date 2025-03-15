import time
import json
from datetime import datetime

class PlayerTracker:
    def __init__(self, player_id, maze_width, maze_height):
        self.player_id = player_id
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.start_time = None
        self.completion_time = None
        self.wrong_turns = 0
        self.backtracks = 0
        self.revisits = 0  # New metric
        self.total_moves = 0  # New metric
        self.path_taken = []
        self.visited_cells = set()
        self.current_x, self.current_y = 1, 1  # Assuming start at (1,1)
    
    def start_tracking(self):
        """Start tracking player movement."""
        self.start_time = time.time()
        self.path_taken.append((self.current_x, self.current_y))
        self.visited_cells.add((self.current_x, self.current_y))
    
    def move(self, direction, maze):
        """Track player movement and update metrics."""
        self.total_moves += 1  # Track all move attempts
        dx, dy = 0, 0
        if direction == 'up':
            dx, dy = 0, -1
        elif direction == 'down':
            dx, dy = 0, 1
        elif direction == 'left':
            dx, dy = -1, 0
        elif direction == 'right':
            dx, dy = 1, 0
        else:
            return False

        new_x, new_y = self.current_x + dx, self.current_y + dy

        if 0 <= new_x < self.maze_width and 0 <= new_y < self.maze_height and maze[new_y][new_x] == 0:
            new_position = (new_x, new_y)
            is_backtrack = False

            # Check for backtrack (previous position)
            if len(self.path_taken) >= 2 and new_position == self.path_taken[-2]:
                self.backtracks += 1
                is_backtrack = True
            elif new_position in self.visited_cells:
                self.revisits += 1  # Count revisits excluding backtracks

            self.current_x, self.current_y = new_x, new_y
            self.path_taken.append(new_position)
            self.visited_cells.add(new_position)

            return True
        else:
            self.wrong_turns += 1
            return False
    
    def complete_maze(self):
        """Record completion time when player reaches the exit."""
        if self.start_time and self.completion_time is None:
            self.completion_time = time.time() - self.start_time
    
    def get_performance_data(self):
        """Return enhanced performance data."""
        return {
            "player_id": self.player_id,
            "maze_size": f"{self.maze_width}x{self.maze_height}",
            "completion_time": self.completion_time,
            "total_moves": self.total_moves,
            "valid_moves": len(self.path_taken) - 1,  # Exclude initial position
            "invalid_moves": self.wrong_turns,
            "backtracks": self.backtracks,
            "revisits": self.revisits,
            "path_length": len(self.path_taken),
            "final_maze_size": f"{self.maze_width}x{self.maze_height}"
        }

    def save_performance_data(self, filename="player_performance.json"):
        """Save performance data to a JSON file."""
        data = self.get_performance_data()
        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []
        existing_data.append(data)
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
