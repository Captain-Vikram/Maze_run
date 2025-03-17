import numpy as np
import random
import os
import heapq
import math
import time
from multiprocessing import Pool, Manager
from adaptivemaze import AdaptiveMazeGame

# Constants
TILE_SIZE = 40
WHITE, BLACK, GREEN, RED, BLUE = (255,) * 3, (0,) * 3, (0, 255, 0), (255, 0, 0), (0, 0, 255)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.2  # Exploration rate
SAVE_INTERVAL = 10  # Save every 10 iterations
SAVE_FOLDER = "Bots"
MAX_BACKTRACKS = 5000  # Max backtracks before regenerating level
MAX_STEPS = 20000  # Max steps before regenerating level
SAVE_FOLDER = "/content/drive/MyDrive/MazeBot_Models"


class QLearningAgent:
    def __init__(self, maze_shape, level):
        self.q_table = np.zeros((*maze_shape, len(ACTIONS)))
        self.visit_counts = {}  # Track state visits
        self.level = level
        self.maze_shape = maze_shape
        self.load_q_table(level, maze_shape)

    def choose_action(self, state, total_steps):
        # Update visit count for current state
        self.visit_counts[state] = self.visit_counts.get(state, 0) + 1

        # Dynamically adjust EPSILON based on total steps and level
        epsilon = max(0.05, 0.3 - (total_steps / 10000) - (self.level * 0.02))

        if random.uniform(0, 1) < epsilon:
            return self.explore_action(state)
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def explore_action(self, state):
        """Smart exploration preferring less-visited paths"""
        valid_actions = []
        for idx, (dx, dy) in enumerate(ACTIONS):
            new_state = (state[0] + dx, state[1] + dy)
            # Delegate validity check to bot class
            if 0 <= new_state[0] < self.q_table.shape[0] and \
               0 <= new_state[1] < self.q_table.shape[1]:
                valid_actions.append((idx, self.visit_counts.get(new_state, 0)))
        
        if valid_actions:
            return min(valid_actions, key=lambda x: x[1])[0]
        return random.choice(range(len(ACTIONS)))

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.max(self.q_table[next_state[0], next_state[1]])
        self.q_table[state[0], state[1], action] += ALPHA * (reward + GAMMA * best_next_action - self.q_table[state[0], state[1], action])
    
    def save_q_table(self, level):
        if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)  # Create directory in Google Drive if it doesn't exist
        filename = os.path.join(SAVE_FOLDER, f"bot_{self.maze_shape[0]}x{self.maze_shape[1]}_lvl_{level}.npy")
        np.save(filename, self.q_table)
        print(f"Saved model at level {level}, grid size {self.maze_shape[0]}x{self.maze_shape[1]}: {filename}")

    def load_q_table(self, current_level, maze_shape):
        try:
            # Look for Q-tables with the exact grid size
            pattern = f"bot_{maze_shape[0]}x{maze_shape[1]}_lvl_"
            available_models = []
            
            if os.path.exists(SAVE_FOLDER):
                for f in os.listdir(SAVE_FOLDER):
                    if f.startswith(pattern) and f.endswith(".npy"):
                        try:
                            level = int(f.split("_lvl_")[1].split(".")[0])
                            available_models.append(level)
                        except:
                            continue
            
            available_models.sort(reverse=True)
            
            if available_models:
                best_level = available_models[0]
                filename = os.path.join(SAVE_FOLDER, f"bot_{maze_shape[0]}x{maze_shape[1]}_lvl_{best_level}.npy")
                self.q_table = np.load(filename)
                print(f"Loaded matching model for {maze_shape[0]}x{maze_shape[1]} grid: {filename}")
                return
                    
            print(f"No model found for grid size {maze_shape[0]}x{maze_shape[1]}. Initializing new Q-table.")
        
        except Exception as e:
            print(f"Error loading Q-table: {str(e)}. Using fresh table.")
            # Initialize a fresh Q-table
            self.q_table = np.zeros((*maze_shape, len(ACTIONS)))

class AStarMazeSolver:
    def __init__(self, game):
        self.game = game
        self.maze = game.maze
        self._validate_start_goal_positions()
        self.reset_stats()
    
    def _validate_start_goal_positions(self):
        """Ensure start/goal positions are within maze bounds"""
        self.start = tuple(np.argwhere(self.game.maze == 2)[0])
        self.goal = tuple(np.argwhere(self.game.maze == 3)[0])
        
        # Clip positions to maze dimensions
        max_y, max_x = self.game.maze.shape
        self.start = (np.clip(self.start[0], 0, max_y-1), 
                     np.clip(self.start[1], 0, max_x-1))
        self.goal = (np.clip(self.goal[0], 0, max_y-1), 
                     np.clip(self.goal[1], 0, max_x-1))
        self.state = self.start
    
    def reset_stats(self):
        """Reset solver statistics"""
        self.path = []
        self.step_count = 0
        self.visited_states = set()
        self.backtrack_count = 0
        self.start_time = time.time()
        self.path_index = 0  # For step-by-step execution
    
    def heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def solve(self):
        """Solve the maze using A* algorithm and return the path"""
        # Priority queue for A*: (f_score, count, node)
        open_set = [(0, 0, self.start)]
        heapq.heapify(open_set)
        
        # For tie-breaking in priority queue
        counter = 0
        
        # Tracking the path
        came_from = {}
        
        # g_score: cost from start to current node
        g_score = {self.start: 0}
        
        # f_score: estimated cost from start to goal through node
        f_score = {self.start: self.heuristic(self.start, self.goal)}
        
        # Set of visited nodes
        closed_set = set()
        
        while open_set:
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            
            # If we've reached the goal, reconstruct and return the path
            if current == self.goal:
                self.path = self._reconstruct_path(came_from, current)
                self.step_count = len(self.path)
                return self.path
            
            # Mark as visited
            closed_set.add(current)
            self.visited_states.add(current)
            
            # Check all possible moves
            for idx, (dx, dy) in enumerate(ACTIONS):
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check if move is valid and not a wall
                if (0 <= neighbor[0] < self.maze.shape[0] and 
                    0 <= neighbor[1] < self.maze.shape[1] and 
                    self.maze[neighbor[0], neighbor[1]] != 1 and
                    neighbor not in closed_set):
                    
                    # Calculate tentative g_score through current node
                    tentative_g_score = g_score.get(current, float('inf')) + 1
                    
                    # If this path is better than previous ones
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        # Update path
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, self.goal)
                        
                        # Add to open set if not there
                        if neighbor not in [item[2] for item in open_set]:
                            counter += 1
                            heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        # No path found
        return []
    
    def _reconstruct_path(self, came_from, current):
        """Reconstruct the path from start to goal"""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return list(reversed(total_path))
    
    def get_action_sequence(self):
        """Convert path to action indices sequence"""
        if not self.path:
            self.solve()
            
        if not self.path:
            return []
            
        actions = []
        for i in range(len(self.path) - 1):
            curr = self.path[i]
            next_pos = self.path[i + 1]
            
            # Calculate direction
            dy = next_pos[0] - curr[0]
            dx = next_pos[1] - curr[1]
            
            # Match to action index
            for idx, (action_dy, action_dx) in enumerate(ACTIONS):
                if action_dy == dy and action_dx == dx:
                    actions.append(idx)
                    break
        
        return actions

class MazeBot:
    def __init__(self, game, level):
        self.agent = QLearningAgent(game.maze.shape, level)
        global EPSILON  # Access the global EPSILON constant
        EPSILON = 0.3  # Increase exploration rate (from 0.2)
        self.game = game
        self._validate_start_goal_positions()
        self.visited_states = set()
        self.backtrack_count = 0
        self.step_count = 0
        self.start_time = time.time()
        self.visited_counts = {}
        self.last_state = None
    
    def step(self):
        if self.backtrack_count >= MAX_BACKTRACKS or self.step_count >= MAX_STEPS:
            print("Too many backtracks/steps. Checking maze solvability...")
            solver = AStarMazeSolver(self.game)
            path = solver.solve()
            if not path:
                print("Unsolvable maze. Regenerating...")
                return "regenerate"
            else:
                print("Maze solvable. Adjusting Q-values.")
                self.agent.q_table *= 0.9  # Reduce overconfidence
                self.backtrack_count = 0
                self.step_count = 0
                return self.state

        action_idx = self.agent.choose_action(self.state, self.step_count)
        action = ACTIONS[action_idx]
        next_state = (self.state[0] + action[0], self.state[1] + action[1])
        
        # Calculate distance-based rewards
        current_dist = self.heuristic(self.state, self.goal)
        new_dist = self.heuristic(next_state, self.goal) if self.is_valid(next_state) else current_dist
        progress_reward = (current_dist - new_dist) * 2  # Amplify progress

        if self.is_valid(next_state):
            reward = 100 if next_state == self.goal else progress_reward
            
            # Dynamic revisit penalty based on visit count
            visit_count = self.visited_counts.get(next_state, 0)
            reward -= 10 * (visit_count + 1)  # Scale penalty with revisits
            
            # Penalize immediate backtrack
            if next_state == self.last_state:
                reward -= 15

            self.visited_counts[next_state] = visit_count + 1
            self.last_state = self.state
            self.state = next_state
        else:
            reward = -20

        # Update Q-table with combined rewards
        self.agent.update_q_table(self.state, action_idx, reward, next_state)
        return self.state
    
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def is_valid(self, state):
        return (0 <= state[0] < self.game.maze.shape[0] and 
                0 <= state[1] < self.game.maze.shape[1] and 
                self.game.maze[state[0], state[1]] != 1)
        
    def get_performance_data(self):
        """Simulate player performance data for difficulty adjustment"""
        return {
            "completion_time": time.time() - self.start_time,
            "total_moves": self.step_count,
            "backtracks": self.backtrack_count,
            "revisits": 0  # Optional: Add revisit tracking if needed
        }
    
    def _validate_start_goal_positions(self):
        """Ensure start/goal positions are within maze bounds"""
        self.start = tuple(np.argwhere(self.game.maze == 2)[0])
        self.goal = tuple(np.argwhere(self.game.maze == 3)[0])
        
        # Clip positions to maze dimensions
        max_y, max_x = self.game.maze.shape
        self.start = (np.clip(self.start[0], 0, max_y-1), 
                      np.clip(self.start[1], 0, max_x-1))
        self.goal = (np.clip(self.goal[0], 0, max_y-1), 
                     np.clip(self.goal[1], 0, max_x-1))
        self.state = self.start

class EnhancedMazeBot(MazeBot):
    def __init__(self, game, level, use_astar_hints=True):
        super().__init__(game, level)
        self.use_astar_hints = use_astar_hints
        self.astar_solver = AStarMazeSolver(game)
        self.optimal_path = []
        self.follow_optimal = False
        self.optimal_index = 0
        self.a_star_cache = {}
        
    def get_optimal_path(self, current_pos):
        """Get optimal path from current position using A*"""
        # Check if we've already calculated this path
        cache_key = (current_pos, self.goal)
        if cache_key in self.a_star_cache:
            return self.a_star_cache[cache_key]
            
        # Update A* solver with current position
        self.astar_solver.start = current_pos
        self.astar_solver.state = current_pos
        self.astar_solver.reset_stats()
        
        # Solve and get action sequence
        path = self.astar_solver.solve()
        actions = self.astar_solver.get_action_sequence()
        
        # Cache the result
        self.a_star_cache[cache_key] = actions

    def step(self):
        # Calculate the backtrack limit based on grid size
        grid_size = self.game.maze.shape[0] * self.game.maze.shape[1]
        max_allowed_backtracks = 5 * math.sqrt(grid_size)
        
        if self.backtrack_count > max_allowed_backtracks or self.step_count >= MAX_STEPS:
            print("Too many backtracks/steps. Checking maze solvability...")
            self.astar_solver.start = self.start
            self.astar_solver.goal = self.goal
            path = self.astar_solver.solve()
            
            if not path:
                print("Unsolvable maze. Regenerating...")
                return "regenerate"
            else:
                print("Maze solvable. Adjusting Q-values.")
                self.agent.q_table *= 0.9
                self.backtrack_count = 0
                self.step_count = 0
                return self.state
        
        # Track dead ends explicitly
        if hasattr(self, 'dead_ends') == False:
            self.dead_ends = set()
            self.current_direction = None
        
        # Get optimal path
        optimal_actions = self.get_optimal_path(self.state)
        
        # Determine whether to follow A* or explore based on how stuck we are
        stuck_factor = self.backtrack_count / max(1, self.step_count)
        follow_astar_prob = min(0.95, 0.7 + stuck_factor * 0.25)  # Increase A* reliance when stuck
        
        if optimal_actions and random.random() < follow_astar_prob:
            # Follow A* path, but avoid dead ends
            for i in range(min(3, len(optimal_actions))):  # Look ahead up to 3 steps
                action_idx = optimal_actions[i]
                next_state = (self.state[0] + ACTIONS[action_idx][0],
                            self.state[1] + ACTIONS[action_idx][1])
                
                # Skip if this leads to a known dead end
                if next_state in self.dead_ends:
                    continue
                    
                if self.is_valid(next_state):
                    # Calculate if this is a direction change
                    if self.current_direction is not None:
                        current_dir = ACTIONS[action_idx]
                        direction_change = current_dir != self.current_direction
                        momentum_factor = 0.8 if not direction_change else 0.5
                    else:
                        momentum_factor = 0.7
                    
                    # Update state
                    reward = 50 * momentum_factor
                    self.agent.update_q_table(self.state, action_idx, reward, next_state)
                    self.last_state = self.state
                    self.current_direction = ACTIONS[action_idx]
                    self.state = next_state
                    self.visited_states.add(self.state)
                    self.step_count += 1
                    
                    # Check if we've reached a dead end
                    valid_moves = self._count_valid_moves(self.state)
                    if valid_moves == 1 and self.state != self.goal:  # Only way out is back
                        self.dead_ends.add(self.state)
                    
                    return self.state
        
        # If not following A*, prioritize unvisited states and avoid dead ends
        valid_actions = []
        for idx, (dy, dx) in enumerate(ACTIONS):
            new_state = (self.state[0] + dy, self.state[1] + dx)
            
            if self.is_valid(new_state) and new_state not in self.dead_ends:
                # Calculate several factors for scoring moves
                is_visited = new_state in self.visited_states
                distance_to_goal = self.heuristic(new_state, self.goal)
                direction_change = (dy, dx) != self.current_direction if self.current_direction else True
                
                # Lower score is better - prioritize unvisited states close to goal in same direction
                score = distance_to_goal * 0.5
                score += 50 if is_visited else 0
                score += 20 if direction_change else 0
                
                valid_actions.append((idx, new_state, score))
        
        if valid_actions:
            # Sort by score (lowest first)
            sorted_actions = sorted(valid_actions, key=lambda x: x[2])
            best_action = sorted_actions[0]
            action_idx = best_action[0]
            next_state = best_action[1]
            
            # Calculate reward based on new vs. revisited state
            if next_state in self.visited_states:
                reward = -30
                if next_state == self.last_state:  # Immediate backtrack
                    reward = -60
                    self.backtrack_count += 1
            else:
                # Reward progress toward goal
                current_dist = self.heuristic(self.state, self.goal)
                new_dist = self.heuristic(next_state, self.goal)
                reward = (current_dist - new_dist) * 15
            
            # Update state
            self.agent.update_q_table(self.state, action_idx, reward, next_state)
            self.visited_states.add(next_state)
            self.last_state = self.state
            self.current_direction = ACTIONS[action_idx]
            self.state = next_state
            self.step_count += 1
            
            # Check if we've reached a dead end
            valid_moves = self._count_valid_moves(self.state)
            if valid_moves == 1 and self.state != self.goal:  # Only way out is back
                self.dead_ends.add(self.state)
        else:
            # No valid moves - we're at a dead end
            self.dead_ends.add(self.state)
            
            # Choose any valid move to escape
            for idx, (dy, dx) in enumerate(ACTIONS):
                new_state = (self.state[0] + dy, self.state[1] + dx)
                if self.is_valid(new_state):
                    self.state = new_state
                    self.step_count += 1
                    return self.state
        
        return self.state

    def _count_valid_moves(self, state):
        """Count how many valid moves are available from this state"""
        count = 0
        for dy, dx in ACTIONS:
            new_state = (state[0] + dy, state[1] + dx)
            if self.is_valid(new_state):
                count += 1
        return count
    
def play_game(use_enhanced=True):
    game = AdaptiveMazeGame("BotPlayer")
    MAX_LEVELS = 100  # Maximum levels to attempt
    level_regen_count = 0  # Track regenerations per level

    while game.current_level <= MAX_LEVELS:
        # Generate maze with current difficulty parameters
        game.generate_maze()
        
        # Choose bot based on flag
        if use_enhanced:
            bot = EnhancedMazeBot(game, game.current_level)
        else:
            bot = MazeBot(game, game.current_level)
        
        # Run the bot until it solves the maze or gets stuck
        result = None
        while bot.state != bot.goal and result != "regenerate":
            result = bot.step()
            
            if result == "regenerate":
                level_regen_count += 1
                if level_regen_count > 3:  # Limit regenerations
                    print(f"Failed to solve level {game.current_level} after 3 attempts.")
                    game.current_level += 1  # Skip to next level
                    level_regen_count = 0
                break
        
        if bot.state == bot.goal:
            # Reward for completing within backtrack quota
            performance_data = bot.get_performance_data()
            game.update_difficulty(performance_data)
            grid_size = game.maze.shape[0] * game.maze.shape[1]
            backtrack_limit = 10 * math.sqrt(grid_size)
            
            # Apply reward for successful completion
            if bot.backtrack_count < backtrack_limit:
                # Amplify positive Q-values as reward
                positive_mask = bot.agent.q_table > 0
                bot.agent.q_table[positive_mask] *= 1.1  # 10% boost as reward
            
            print(f"Level {game.current_level} solved! Skill: {game.difficulty}")
            
            # Save progress periodically
            if game.current_level % SAVE_INTERVAL == 0:
                bot.agent.save_q_table(game.current_level)
                
            # Reset counters for next level
            game.current_level += 1
            level_regen_count = 0
            
        # Reset shape changed flag
        game.maze_shape_changed = False

from multiprocessing import Pool, cpu_count

def train_grid_size(grid_size):
    """
    Train the bot on a specific grid size.
    This function will be executed in parallel by each worker.
    """
    print(f"Training on grid size: {grid_size[0]}x{grid_size[1]}")
    game = AdaptiveMazeGame("BotPlayer")
    game.maze_params = {"width": grid_size[0], "height": grid_size[1]}
    current_level = 1
    performance_metrics = []

    # Train the bot on 100 maps of the same grid size
    for iteration in range(10000):
        print(f"Iteration {iteration + 1}/10000 for grid size {grid_size[0]}x{grid_size[1]}")
        while True:
            game.generate_maze()
            solver = AStarMazeSolver(game)
            path = solver.solve()
            if len(path) > 0:
                break
            print("Unsolvable maze detected, regenerating...")

        print(f"Generated new maze for grid size: {grid_size[0]}x{grid_size[1]}")
        bot = EnhancedMazeBot(game, level=current_level)

        result = None
        while bot.state != bot.goal and result != "regenerate":
            result = bot.step()
            if result == "regenerate":
                game.generate_maze()
                bot = MazeBot(game, game.current_level)
                print("Maze regenerated.")

        if bot.state == bot.goal:
            performance_data = bot.get_performance_data()
            performance_metrics.append(performance_data)
            print(f"Maze solved! Moves: {performance_data['total_moves']}, Backtracks: {performance_data['backtracks']}")

        if iteration % 1 == 0:
            bot.agent.save_q_table(current_level)
            current_level += 1

    # Calculate average performance metrics for this grid size
    if performance_metrics:
        avg_completion_time = np.mean([data["completion_time"] for data in performance_metrics])
        avg_total_moves = np.mean([data["total_moves"] for data in performance_metrics])
        avg_backtracks = np.mean([data["backtracks"] for data in performance_metrics])
        print(f"Average Performance for Grid Size {grid_size[0]}x{grid_size[1]}:")
        print(f"  Completion Time: {avg_completion_time:.2f}s")
        print(f"  Total Moves: {avg_total_moves:.2f}")
        print(f"  Backtracks: {avg_backtracks:.2f}")
    else:
        print(f"No successful completions for grid size {grid_size[0]}x{grid_size[1]}")

    return {
        "grid_size": grid_size,
        "avg_completion_time": avg_completion_time if performance_metrics else None,
        "avg_total_moves": avg_total_moves if performance_metrics else None,
        "avg_backtracks": avg_backtracks if performance_metrics else None,
    }

def play_game_with_training():
    """
    Train bots on multiple grid sizes using multiprocessing.
    """
    # Define grid sizes to train on
    grid_sizes = [(x, x) for x in range(19, 31)]  # 11x11 to 30x30

    # Use multiprocessing to parallelize grid size training
    num_workers = cpu_count()  # Number of CPU cores available
    with Pool(processes=num_workers) as pool:
        results = pool.map(train_grid_size, grid_sizes)

    # Collect and display results
    print("\nTraining Summary:")
    for result in results:
        grid_size = result["grid_size"]
        avg_completion_time = result["avg_completion_time"]
        avg_total_moves = result["avg_total_moves"]
        avg_backtracks = result["avg_backtracks"]

        print(f"\nGrid Size: {grid_size[0]}x{grid_size[1]}")
        if avg_completion_time is not None:
            print(f"  Average Completion Time: {avg_completion_time:.2f}s")
            print(f"  Average Total Moves: {avg_total_moves:.2f}")
            print(f"  Average Backtracks: {avg_backtracks:.2f}")
        else:
            print("  No successful completions.")

    print("\nTraining completed for all grid sizes.")

if __name__ == "__main__":
    play_game_with_training()