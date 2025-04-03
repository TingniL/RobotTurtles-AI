import logging
import numpy as np
from typing import Dict, Any
from ..environment.game_env import RobotTurtlesEnv

class GameVisualizer:
    """Visualizer for Robot Turtles game (simplified version)"""
    
    def __init__(self):
        """Initialize the visualizer"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing simplified game visualizer...")
        
        # Initialize visualization settings
        self.board_size = 8
        self.cell_size = 60
        self.margin = 20
        
        self.logger.info("Simplified visualizer initialized successfully")
        
    def render(self, board: np.ndarray, info: Dict[str, Any] = None) -> None:
        """Log game state instead of rendering (OpenCV-free version)
        
        Args:
            board: Game board state
            info: Additional game information
        """
        self.logger.info("Logging game state (rendering disabled)...")
        
        if info:
            self.logger.info(f"Game info: Steps: {info.get('steps_taken', 0)}, Player position: {info.get('player_pos', 'unknown')}")
        
        # Count elements on board for logging
        player_count = np.sum(board[0])
        wall_count = np.sum(board[1]) if board.shape[0] > 1 else 0
        
        self.logger.info(f"Board state: {board.shape}, Players: {player_count}, Walls: {wall_count}")
        
    def close(self) -> None:
        """Close visualization (no-op in simplified version)"""
        self.logger.info("Closing visualization (no-op in simplified version)")

    def render_game_state(self, env: RobotTurtlesEnv, figsize=(10, 10)):
        """Render the game state using text logging only
        
        Args:
            env: Game environment to render
            figsize: Size of figure (ignored in this version)
        """
        self.logger.info("Rendering game state as text...")
        self.logger.info(f"Player position: {env.player_pos if hasattr(env, 'player_pos') else 'unknown'}")
        self.logger.info(f"Steps taken: {env.steps_taken if hasattr(env, 'steps_taken') else 'unknown'}")
        self.logger.info(f"Game completed: {env.done if hasattr(env, 'done') else 'unknown'}")
        self.logger.info("Game state rendered as text") 