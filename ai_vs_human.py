import pygame
import time
import random
import numpy as np
import os
from adaptivemaze import AdaptiveMazeGame

# Constants
TILE_SIZE = 40
MAX_FPS = 120
STATS_WIDTH = 250
MAX_WINDOW_SIZE = (1000, 700)
WHITE, BLACK, GREEN, RED, BLUE, YELLOW = (255,)*3, (0,)*3, (0,255,0), (255,0,0), (0,0,255), (255,255,0)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
SAVE_FOLDER = "Bots"

class QLearningAgent:
    def __init__(self, maze_shape, level):
        self.q_table = np.zeros((*maze_shape, len(ACTIONS)))
        self.load_q_table(level)
    
    def choose_action(self, state):
        # Epsilon-greedy action selection with reduced exploration for better gameplay
        if random.uniform(0, 1) < 0.1:
            return random.choice(range(len(ACTIONS)))
        else:
            return np.argmax(self.q_table[state[0], state[1]])
    
    def update_q_table(self, state, action, reward, next_state):
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        best_next_action = np.max(self.q_table[next_state[0], next_state[1]])
        self.q_table[state[0], state[1], action] += alpha * (reward + gamma * best_next_action - self.q_table[state[0], state[1], action])
    
    def save_q_table(self, level):
        if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)
        filename = os.path.join(SAVE_FOLDER, f"bot_lvl_{level}.npy")
        np.save(filename, self.q_table)
        print(f"Saved model at level {level}: {filename}")
    
    def load_q_table(self, current_level):
        """Load the appropriate trained model based on current level."""
        if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)
            # No models available, just use default
            if os.path.exists("q_table.npy"):
                self.q_table = np.load("q_table.npy")
                print("Loaded default Q-table.")
            return
            
        available_models = []
        try:
            available_models = sorted(
                [int(f.split("_")[2].split(".")[0]) for f in os.listdir(SAVE_FOLDER) if f.startswith("bot_lvl_")],
                reverse=True
            )
        except:
            pass
        
        chosen_level = None
        for lvl in available_models:
            if int(current_level) >= lvl:
                chosen_level = lvl
                break
        
        if chosen_level:
            filename = os.path.join(SAVE_FOLDER, f"bot_lvl_{chosen_level}.npy")
            try:
                self.q_table = np.load(filename)
                print(f"Loaded model: {filename}")
            except:
                print(f"Failed to load {filename}, using default Q-table")
                if os.path.exists("q_table.npy"):
                    self.q_table = np.load("q_table.npy")
        elif os.path.exists("q_table.npy"):
            self.q_table = np.load("q_table.npy")
            print("Loaded default Q-table.")

class AStarSolver:
    def __init__(self, maze):
        self.maze = maze
    
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def solve(self, start, goal):
        import heapq
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            for dx, dy in ACTIONS:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.maze.shape[0] and 0 <= neighbor[1] < self.maze.shape[1] and self.maze[neighbor] != 1:
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []
    
class MazeBot:
    def __init__(self, game, level):
        self.agent = QLearningAgent(game.maze.shape, level)
        self.solver = AStarSolver(game.maze)
        self.game = game
        self.start = tuple(np.argwhere(game.maze == 2)[0])
        self.goal = tuple(np.argwhere(game.maze == 3)[0])
        self.state = self.start
        self.path = [self.start]
        self.solution_path = self.solver.solve(self.start, self.goal)
        self.solution_index = 0
        self.total_moves = 0
        self.backtracks = 0
        self.complete = False
        self.thinking_time = random.uniform(0.3, 0.7)  # Bot "thinks" before moving
        self.last_move_time = time.time()
        self.difficulty_level = 1  # Adjusts bot speed
    
    def step(self, use_solution=True):
        """Move bot step-by-step while learning."""
        current_time = time.time()
        
        # Add thinking delay based on difficulty
        if current_time - self.last_move_time < self.thinking_time / self.difficulty_level:
            return self.state
            
        if self.state == self.goal:
            self.complete = True
            return self.state
            
        # Use A* solution or Q-learning based on parameter
        if use_solution and self.solution_path and self.solution_index < len(self.solution_path):
            next_state = self.solution_path[self.solution_index]
            self.solution_index += 1
        else:
            action_idx = self.agent.choose_action(self.state)
            action = ACTIONS[action_idx]
            next_state = (self.state[0] + action[0], self.state[1] + action[1])

            # Check if valid move
            if not (0 <= next_state[0] < self.game.maze.shape[0] and 
                   0 <= next_state[1] < self.game.maze.shape[1] and 
                   self.game.maze[next_state] != 1):
                return self.state  # Invalid move
                
            # Update Q-table
            reward = 100 if next_state == self.goal else -1
            self.agent.update_q_table(self.state, action_idx, reward, next_state)
        
        # Update bot state and tracking info
        if next_state in self.path:
            self.backtracks += 1
            
        self.path.append(next_state)
        self.total_moves += 1
        self.state = next_state
        self.last_move_time = current_time
        
        return next_state
        
    def set_difficulty(self, level):
        """Adjust bot difficulty from 1-5"""
        if isinstance(level, str):  # Convert difficulty levels to numeric values
            difficulty_map = {"beginner": 1, "intermediate": 3, "advanced": 5}
            level = difficulty_map.get(level.lower(), 1)  # Default to 1 if not found

        self.difficulty_level = max(1, min(5, level))
        self.thinking_time = random.uniform(0.6, 1.0) - (self.difficulty_level * 0.1)

class AIHumanMazeUI:
    def __init__(self):
        pygame.init()
        self.game = AdaptiveMazeGame("DualPlayer")
        self.running = True
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.title_font = pygame.font.Font(None, 36)
        self.game_over_font = pygame.font.Font(None, 56)
        self.load_new_level()  # Initialize the first level
        self.match_stats = {"ai_wins": 0, "human_wins": 0, "games": 0}
        self.game_state = "playing"  # "playing", "human_won", "ai_won"

    def init_ai(self):
        """Initialize AI game instance and bot."""
        self.ai_game = AdaptiveMazeGame("BotPlayer")
        self.ai_game.generate_maze()
        self.bot = MazeBot(self.ai_game, level=self.ai_game.difficulty)
        self.bot.set_difficulty(self.ai_game.difficulty)

    def init_human(self):
        """Initialize Human game instance and player."""
        self.human_game = AdaptiveMazeGame("HumanPlayer")
        self.human_game.generate_maze()
        self.player_pos = np.argwhere(self.human_game.maze == 2)[0].astype(float)
        self.player_path = [tuple(self.player_pos.astype(int))]
        self.player_moves = 0
        self.player_backtracks = 0
        self.player_complete = False
        self.start_time = time.time()

    def load_new_level(self):
        """Generate new level with dynamic window sizing."""
        self.game.generate_maze()
        self.init_ai()  # Initialize AI game and bot
        self.init_human()  # Initialize human game and player

        self.maze = self.game.maze
        self.height, self.width = self.maze.shape

        # Calculate required window size
        maze_pixel_width = self.width * TILE_SIZE
        maze_pixel_height = self.height * TILE_SIZE

        # Apply window size constraints
        window_width = min(maze_pixel_width * 2 + STATS_WIDTH, MAX_WINDOW_SIZE[0])
        window_height = min(maze_pixel_height, MAX_WINDOW_SIZE[1])

        # Set up display
        self.window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption(f"AI vs Human Maze Race - Level {self.game.current_level}")

        # Track time
        self.start_time = time.time()
        self.player_complete = False
        self.game_state = "playing"

        # Path colors
        self.player_color = (0, 200, 255)  # Cyan-ish
        self.bot_color = (255, 100, 100)  # Red-ish

    def calculate_camera(self, pos, maze):
        """Calculate camera position based on maze dimensions."""
        viewport_width = (self.window.get_width() - STATS_WIDTH) // 2
        viewport_height = self.window.get_height()

        cam_x = pos[1] * TILE_SIZE - viewport_width // 2
        cam_y = pos[0] * TILE_SIZE - viewport_height // 2

        max_cam_x = max(0, maze.shape[1] * TILE_SIZE - viewport_width)
        max_cam_y = max(0, maze.shape[0] * TILE_SIZE - viewport_height)

        return np.clip(cam_x, 0, max_cam_x), np.clip(cam_y, 0, max_cam_y)

    def draw_maze(self, maze, pos, offset_x, is_ai=False, path=None):
        """Draw a specific maze (AI or Human) with offset."""
        cam_x, cam_y = self.calculate_camera(pos, maze)
        viewport_width = (self.window.get_width() - STATS_WIDTH) // 2

        if is_ai:
            title = "AI Bot"
            path_color = self.bot_color
            current_pos = self.bot.state
        else:
            title = "Player"
            path_color = self.player_color
            current_pos = tuple(self.player_pos.astype(int))

        # Background
        bg = pygame.transform.scale(
            pygame.image.load("assets/grass.jpeg"),
            (maze.shape[1] * TILE_SIZE, maze.shape[0] * TILE_SIZE)
        )
        self.window.blit(bg, (offset_x - cam_x, -cam_y))

        # Draw path
        if path:
            for p in path:
                if p != current_pos:  # Don't draw current position
                    px = p[1] * TILE_SIZE - cam_x + offset_x
                    py = p[0] * TILE_SIZE - cam_y
                    path_rect = pygame.Rect(px + 10, py + 10, TILE_SIZE - 20, TILE_SIZE - 20)
                    pygame.draw.rect(self.window, path_color, path_rect, border_radius=5)

        # Load assets
        try:
            wall = pygame.transform.scale(pygame.image.load("assets/wall.jpeg"), (TILE_SIZE, TILE_SIZE))
            exit_img = pygame.transform.scale(pygame.image.load("assets/finish.png"), (TILE_SIZE, TILE_SIZE))
            player = pygame.transform.scale(pygame.image.load("assets/player.png"), (TILE_SIZE, TILE_SIZE))
            bot = pygame.transform.scale(pygame.image.load("assets/bot.png"), (TILE_SIZE, TILE_SIZE))
        except FileNotFoundError:
            wall = None
            exit_img = None
            player = None
            bot = None

        # Draw maze elements
        for row in range(maze.shape[0]):
            for col in range(maze.shape[1]):
                x = col * TILE_SIZE - cam_x + offset_x
                y = row * TILE_SIZE - cam_y

                # Culling off-screen tiles
                if not (-TILE_SIZE <= x <= viewport_width + TILE_SIZE and
                        -TILE_SIZE <= y <= self.window.get_height() + TILE_SIZE):
                    continue

                if maze[row, col] == 1:  # Wall
                    if wall:
                        self.window.blit(wall, (x, y))
                    else:
                        pygame.draw.rect(self.window, (80, 80, 80), (x, y, TILE_SIZE, TILE_SIZE))

                elif maze[row, col] == 3:  # Exit
                    if exit_img:
                        self.window.blit(exit_img, (x, y))
                    else:
                        pygame.draw.rect(self.window, GREEN, (x, y, TILE_SIZE, TILE_SIZE))

                    # Add glow effect for exit
                    glow = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                    glow.fill((0, 255, 0, 100))
                    self.window.blit(glow, (x, y), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw player/bot
        px = pos[1] * TILE_SIZE - cam_x + offset_x
        py = pos[0] * TILE_SIZE - cam_y

        if is_ai:
            if bot:
                self.window.blit(bot, (px, py))
            else:
                pygame.draw.rect(self.window, RED, (px + 5, py + 5, TILE_SIZE - 10, TILE_SIZE - 10), border_radius=10)
        else:
            if player:
                self.window.blit(player, (px, py))
            else:
                pygame.draw.rect(self.window, BLUE, (px + 5, py + 5, TILE_SIZE - 10, TILE_SIZE - 10), border_radius=10)

        # Draw title
        title_bg = pygame.Rect(offset_x, 0, viewport_width, 40)
        pygame.draw.rect(self.window, (30, 30, 30), title_bg)
        title_text = self.title_font.render(title, True, WHITE)
        self.window.blit(title_text, (offset_x + viewport_width // 2 - title_text.get_width() // 2, 10))

    def draw_stats_panel(self):
        """Draw an enhanced game statistics panel in the center."""
        # Define panel properties
        panel_x = (self.window.get_width() - STATS_WIDTH) // 2
        panel_height = self.window.get_height()
        panel_bg_color = (40, 40, 40)  # Dark gray background
        panel_border_color = (200, 200, 200)  # Light gray border
        title_color = WHITE
        stat_label_color = (150, 150, 150)  # Light gray for labels
        stat_value_color = WHITE  # Bright white for values

        # Draw panel background with rounded corners
        pygame.draw.rect(self.window, panel_bg_color, (panel_x, 0, STATS_WIDTH, panel_height), border_radius=10)
        pygame.draw.rect(self.window, panel_border_color, (panel_x, 0, STATS_WIDTH, panel_height), 3, border_radius=10)

        # Game Title
        title = self.title_font.render("AI vs Human Maze Race", True, title_color)
        self.window.blit(title, (panel_x + STATS_WIDTH // 2 - title.get_width() // 2, 20))

        # Current match stats
        elapsed_time = time.time() - self.start_time
        level_text = self.font.render(f"Level: {self.human_game.current_level}", True, stat_value_color)
        diff_text = self.font.render(f"Difficulty: {self.ai_game.difficulty}/5", True, stat_value_color)
        time_text = self.font.render(f"Time: {elapsed_time:.1f}s", True, stat_value_color)

        # AI Bot Stats
        ai_moves = self.font.render(f"Moves: {self.bot.total_moves}", True, stat_value_color)
        ai_backtracks = self.font.render(f"Backtracks: {self.bot.backtracks}", True, stat_value_color)
        ai_complete = self.font.render(f"Complete: {'Yes' if self.bot.complete else 'No'}", True, GREEN if self.bot.complete else RED)

        # Player Stats
        player_moves = self.font.render(f"Moves: {self.player_moves}", True, stat_value_color)
        player_backtracks = self.font.render(f"Backtracks: {self.player_backtracks}", True, stat_value_color)
        player_complete = self.font.render(f"Complete: {'Yes' if self.player_complete else 'No'}", True, GREEN if self.player_complete else RED)

        # Match Score
        match_score = self.font.render(f"AI Wins: {self.match_stats['ai_wins']} | Human Wins: {self.match_stats['human_wins']}", True, stat_value_color)

        # Render stats with proper spacing and alignment
        stats = [
            ("Match Stats", 70, title_color, True),  # Section header
            (level_text, 110, stat_value_color),
            (diff_text, 150, stat_value_color),
            (time_text, 190, stat_value_color),
            ("AI Bot Stats", 250, title_color, True),  # Section header
            (ai_moves, 290, stat_value_color),
            (ai_backtracks, 330, stat_value_color),
            (ai_complete, 370, GREEN if self.bot.complete else RED),
            ("Player Stats", 430, title_color, True),  # Section header
            (player_moves, 470, stat_value_color),
            (player_backtracks, 510, stat_value_color),
            (player_complete, 550, GREEN if self.player_complete else RED),
            ("Match Score", 610, title_color, True),  # Section header
            (match_score, 650, stat_value_color),
        ]

        for entry in stats:
            # Unpack the entry with a default value for is_header
            if len(entry) == 4:
                text_surface, ypos, color, is_header = entry
            else:
                text_surface, ypos, color = entry
                is_header = False  # Default value if is_header is not provided

            # Handle section headers (strings)
            if isinstance(text_surface, str):
                text_surface = self.font.render(text_surface, True, color)
                x_pos = panel_x + STATS_WIDTH // 2 - text_surface.get_width() // 2
            else:
                x_pos = panel_x + 20

            # Draw the text surface
            self.window.blit(text_surface, (x_pos, ypos))

    def draw_separators(self):
        """Draw visual separators between AI, stats, and human panels."""
        panel_x = (self.window.get_width() - STATS_WIDTH) // 2
        separator_color = (100, 100, 100)
        # Vertical separator between AI and stats
        pygame.draw.line(self.window, separator_color, (panel_x, 0), (panel_x, self.window.get_height()), 3)
        # Vertical separator between stats and human
        pygame.draw.line(self.window, separator_color, (panel_x + STATS_WIDTH, 0), (panel_x + STATS_WIDTH, self.window.get_height()), 3)

    def draw(self):
        """Draw both AI and Human mazes side by side with stats in the center."""
        self.window.fill(BLACK)
        viewport_width = (self.window.get_width() - STATS_WIDTH) // 2
        # Draw Human maze on the left
        self.draw_maze(
            self.human_game.maze,
            self.player_pos,
            0,
            is_ai=False,
            path=self.player_path
        )
        # Draw AI maze on the right
        self.draw_maze(
            self.ai_game.maze,
            self.bot.state,
            viewport_width + STATS_WIDTH,
            is_ai=True,
            path=self.bot.path
        )
        # Draw stats panel in the center
        self.draw_stats_panel()
        # Draw separators
        self.draw_separators()
        # Draw game over screen if needed
        self.draw_game_over()
        # Update display
        pygame.display.flip()

    def run(self):
        """Main game loop."""
        while self.running:
            self.handle_events()
            self.update_ai()
            self.draw()
            self.clock.tick(MAX_FPS)
        # Save AI learning before exit
        self.bot.agent.save_q_table()
        pygame.quit()
if __name__ == "__main__":
    game = AIHumanMazeUI()  # Use the correct class name
    game.run()  # Start the game loop
