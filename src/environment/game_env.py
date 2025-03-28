import logging
from typing import Tuple, Dict, Any
import numpy as np

class RobotTurtlesEnv:
    """Robot Turtles game environment"""
    
    def __init__(self, config: GameConfig):
        """Initialize the environment
        
        Args:
            config: Game configuration
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Robot Turtles environment...")
        
        self.config = config
        self.board_size = 8
        self.max_steps = 20  # Limit maximum steps to prevent infinite loops
        
        # Initialize game state
        self.reset()
        
        self.logger.info("Environment initialized successfully")
        
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state
        
        Returns:
            Tuple of (observation, info)
        """
        self.logger.info("Resetting environment...")
        
        # Initialize board
        self.board = np.zeros((9, self.board_size, self.board_size), dtype=np.float32)
        
        # Initialize player position
        self.player_pos = (0, 0)  # Start at top-left corner
        self.board[0, 0, 0] = 1  # Mark player position
        
        # Initialize game state
        self.steps_taken = 0
        self.done = False
        self.info = {}
        
        # Get initial observation
        obs = self._get_observation()
        
        self.logger.info("Environment reset completed")
        return obs, self.info
        
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment
        
        Args:
            action: Action dictionary containing action_type and wall_position
            
        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        self.logger.info(f"Executing step {self.steps_taken + 1}/{self.max_steps}")
        
        # Check if game is already done
        if self.done:
            self.logger.warning("Game is already done, returning current state")
            return self._get_observation(), 0.0, True, False, self.info
            
        # Increment step counter
        self.steps_taken += 1
        
        # Get action components
        action_type = action['action_type']
        wall_pos = action['wall_position']
        
        # Apply action
        if action_type == 0:  # Move up
            new_pos = (max(0, self.player_pos[0] - 1), self.player_pos[1])
        elif action_type == 1:  # Move right
            new_pos = (self.player_pos[0], min(self.board_size - 1, self.player_pos[1] + 1))
        elif action_type == 2:  # Move down
            new_pos = (min(self.board_size - 1, self.player_pos[0] + 1), self.player_pos[1])
        elif action_type == 3:  # Move left
            new_pos = (self.player_pos[0], max(0, self.player_pos[1] - 1))
        else:  # Place wall
            if 0 <= wall_pos[0] < self.board_size and 0 <= wall_pos[1] < self.board_size:
                self.board[1, wall_pos[0], wall_pos[1]] = 1
            new_pos = self.player_pos
            
        # Update player position
        self.board[0, self.player_pos[0], self.player_pos[1]] = 0
        self.board[0, new_pos[0], new_pos[1]] = 1
        self.player_pos = new_pos
        
        # Check win condition
        self.done = self._check_win_condition()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if maximum steps reached
        truncated = self.steps_taken >= self.max_steps
        
        # Update info
        self.info = {
            'steps_taken': self.steps_taken,
            'player_pos': self.player_pos,
            'done': self.done,
            'truncated': truncated
        }
        
        self.logger.info(f"Step completed: reward={reward:.2f}, done={self.done}, truncated={truncated}")
        return self._get_observation(), reward, self.done, truncated, self.info
        
    def _check_win_condition(self) -> bool:
        """Check if the game is won
        
        Returns:
            bool: True if game is won, False otherwise
        """
        # Win if player has moved from starting position
        return self.player_pos != (0, 0)
        
    def _calculate_reward(self) -> float:
        """Calculate reward for current state
        
        Returns:
            float: Reward value
        """
        # Small penalty for each step to encourage faster completion
        return -0.1 if not self.done else 1.0
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation
        
        Returns:
            Dict containing observation arrays
        """
        return {
            'board': self.board.copy(),
            'hand': np.zeros((5, 5), dtype=np.float32),  # Simplified hand representation
            'program': np.zeros((self.config.max_program_size, 5), dtype=np.float32)  # Simplified program representation
        }
        
    def render(self) -> None:
        """Render the current game state"""
        self.logger.info("Rendering game state...")
        # Implementation of rendering logic
        pass
        
    def close(self) -> None:
        """Close the environment"""
        self.logger.info("Closing environment...")
        # Clean up resources if needed
        pass

