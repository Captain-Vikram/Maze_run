import pygame
import time
import random
import numpy as np
from adaptivemaze import AdaptiveMazeGame

# Constants
TILE_SIZE = 40
MAX_FPS = 120  # Maintain high FPS for smooth animations
STATS_WIDTH = 250
MAX_WINDOW_SIZE = (800, 600)  # Maximum window dimensions
WHITE, BLACK, GREEN, RED, BLUE = (255,)*3, (0,)*3, (0,255,0), (255,0,0), (0,0,255)

class MazeGameUI:
    def __init__(self, player_id="Player1"):
        pygame.init()
        self.game = AdaptiveMazeGame(player_id)
        self.running = True
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.path = []
        self.load_new_level()

    def load_new_level(self):
        """Generate new level with dynamic window sizing"""
        self.game.generate_maze()
        self.player_tracker = self.game.create_player_tracker()
        self.player_tracker.start_tracking()
        
        self.maze = self.game.maze
        self.height, self.width = self.maze.shape
        
        # Player starts at entry point (where maze value is 2)
        self.player_pos = np.argwhere(self.maze == 2)[0].astype(float)
        
        # Calculate required window size
        maze_pixel_width = self.width * TILE_SIZE
        maze_pixel_height = self.height * TILE_SIZE
        
        # Apply window size constraints
        window_width = min(maze_pixel_width + STATS_WIDTH, MAX_WINDOW_SIZE[0])
        window_height = min(maze_pixel_height, MAX_WINDOW_SIZE[1])
        
        # Set up display
        self.window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption(f"Adaptive Maze - Level {self.game.current_level}")
        self.start_time = time.time()
        self.path = [tuple(self.player_pos.astype(int))]

    def calculate_camera(self):
        """Dynamic camera system with boundary checks"""
        viewport_width = self.window.get_width() - STATS_WIDTH
        viewport_height = self.window.get_height()
        
        # Calculate base camera position
        cam_x = self.player_pos[1] * TILE_SIZE - viewport_width // 2
        cam_y = self.player_pos[0] * TILE_SIZE - viewport_height // 2
        
        # Clamp to maze boundaries
        max_cam_x = max(0, self.width * TILE_SIZE - viewport_width)
        max_cam_y = max(0, self.height * TILE_SIZE - viewport_height)
        
        return (
            np.clip(cam_x, 0, max_cam_x),
            np.clip(cam_y, 0, max_cam_y)
        )

    def draw_maze(self):
        """Render maze with textures and smooth camera"""
        cam_x, cam_y = self.calculate_camera()
        viewport_width = self.window.get_width() - STATS_WIDTH
        
        # Background
        bg = pygame.transform.scale(
            pygame.image.load("assets/grass.jpeg"),
            (self.width*TILE_SIZE, self.height*TILE_SIZE)
        )
        self.window.blit(bg, (-cam_x, -cam_y))
        
        # Load assets
        wall = pygame.transform.scale(pygame.image.load("assets/wall.jpeg"), (TILE_SIZE, TILE_SIZE))
        exit_img = pygame.transform.scale(pygame.image.load("assets/finish.png"), (TILE_SIZE, TILE_SIZE))
        player = pygame.transform.scale(pygame.image.load("assets/player.png"), (TILE_SIZE, TILE_SIZE))
        start_img = pygame.transform.scale(pygame.image.load("assets/start.png"), (TILE_SIZE, TILE_SIZE))  # Load start image

        # Draw maze elements
        for row in range(self.height):
            for col in range(self.width):
                x = col*TILE_SIZE - cam_x
                y = row*TILE_SIZE - cam_y
                
                # Culling off-screen tiles
                if not (-TILE_SIZE <= x <= viewport_width + TILE_SIZE and 
                        -TILE_SIZE <= y <= self.window.get_height() + TILE_SIZE):
                    continue
                
                if self.maze[row, col] == 1:
                    self.window.blit(wall, (x, y))
                elif self.maze[row, col] == 3:
                    self.window.blit(exit_img, (x, y))
                    glow = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                    glow.fill((0, 255, 0, 100))
                    self.window.blit(glow, (x, y), special_flags=pygame.BLEND_RGBA_ADD)
                elif self.maze[row, col] == 2:  # Draw start position
                    self.window.blit(start_img, (x, y))

        # Draw player
        px = self.player_pos[1]*TILE_SIZE - cam_x
        py = self.player_pos[0]*TILE_SIZE - cam_y
        self.window.blit(player, (px, py))

        # Draw stats panel
        panel_x = viewport_width
        pygame.draw.rect(self.window, (30,30,30), (panel_x, 0, STATS_WIDTH, self.window.get_height()))
        pygame.draw.rect(self.window, (200,200,200), (panel_x, 0, STATS_WIDTH, self.window.get_height()), 3)
        
        # Display statistics
        stats = [
            (f"Level {self.game.current_level}", 50),
            (f"Time: {time.time()-self.start_time:.1f}s", 100),
            (f"Moves: {self.player_tracker.total_moves}", 150),
            (f"Backtracks: {self.player_tracker.backtracks}", 200),
            (f"Difficulty: {self.game.difficulty}", 250)
        ]
        
        for text, ypos in stats:
            self.window.blit(self.font.render(text, True, WHITE), (panel_x+20, ypos))


    def move_player(self, dx, dy):
        """Smooth movement with backtracking detection"""
        new_pos = self.player_pos + [dx, dy]
        new_x, new_y = new_pos.astype(int)

        if (0 <= new_x < self.height and 
            0 <= new_y < self.width and 
            self.maze[new_x, new_y] != 1):
            
            # Backtrack detection
            current = (new_x, new_y)
            if current in self.path[:-1]:
                self.player_tracker.backtracks += 1

            # Smooth animation (more frames)
            prev_pos = self.player_pos.copy()
            for t in np.linspace(0, 1, 5):  # More steps for smooth movement
                self.player_pos = prev_pos + (dx * t, dy * t)
                self.draw_maze()
                pygame.display.flip()
                self.clock.tick(60)  # Increase FPS for smoother animation

            # Finalize position
            self.player_pos = new_pos
            self.path.append(current)
            self.player_tracker.total_moves += 1

            # Check level completion
            if self.maze[new_x, new_y] == 3:
                self.complete_level()

    def complete_level(self):
        """Progress to next level with difficulty adjustment"""
        self.player_tracker.complete_maze()
        self.game.update_difficulty(self.player_tracker.get_performance_data())
        self.load_new_level()

    def handle_events(self):
        """Process keyboard input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.move_player(-1, 0)
                elif event.key == pygame.K_DOWN:
                    self.move_player(1, 0)
                elif event.key == pygame.K_LEFT:
                    self.move_player(0, -1)
                elif event.key == pygame.K_RIGHT:
                    self.move_player(0, 1)

    def run(self):
        """Main game loop with FPS control"""
        while self.running:
            self.handle_events()
            self.draw_maze()
            pygame.display.flip()
            self.clock.tick(MAX_FPS)
        pygame.quit()

if __name__ == "__main__":
    game = MazeGameUI()
    game.run()