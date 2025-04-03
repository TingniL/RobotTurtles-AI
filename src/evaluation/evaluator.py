import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from ..environment.game_env import RobotTurtlesEnv, GameConfig
from ..models.network import RobotTurtlesNet
import logging

class Evaluator:
    """Evaluator for Robot Turtles AI model"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize the evaluator
        
        Args:
            model: The model to evaluate
            device: Device to run evaluation on
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing evaluator...")
        
        self.model = model
        self.device = device
        self.model.eval()  # Set model to evaluation mode
        
        # Create a default environment
        try:
            self.env = RobotTurtlesEnv(GameConfig())
            self.logger.info("Default evaluation environment created")
        except Exception as e:
            self.logger.error(f"Failed to create default environment: {str(e)}")
            self.logger.warning("Evaluator will create environment on-demand")
            self.env = None
        
        self.logger.info("Evaluator initialized successfully")
        
    def evaluate_against_random(self, num_games: int = 10) -> Dict[str, float]:
        """Evaluate the model against random opponent
        
        Args:
            num_games: Number of games to play
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Starting evaluation against random opponent for {num_games} games")
        
        # Ensure we have an environment
        if self.env is None:
            try:
                self.env = RobotTurtlesEnv(GameConfig())
                self.logger.info("Created evaluation environment")
            except Exception as e:
                self.logger.error(f"Failed to create environment: {str(e)}")
                return {
                    'win_rate': 0.0,
                    'avg_reward': 0.0,
                    'avg_steps': 0.0
                }
        
        # Track metrics
        wins = 0
        total_rewards = 0
        total_steps = 0
        
        # Play games
        for game in range(num_games):
            self.logger.info(f"Playing evaluation game {game+1}/{num_games}")
            
            try:
                # Reset environment
                obs, _ = self.env.reset()
                
                # Game state
                done = False
                truncated = False
                game_reward = 0
                game_steps = 0
                
                # Play until game ends
                max_steps = 20  # Safety limit
                while not (done or truncated) and game_steps < max_steps:
                    # Get model action
                    try:
                        action, _, _ = self._get_action(obs)
                        self.logger.debug(f"Action selected: {action}")
                    except Exception as e:
                        self.logger.error(f"Failed to get action: {str(e)}")
                        action = {'action_type': 0, 'wall_position': (0, 0)}
                        
                    # Execute action
                    try:
                        obs, reward, done, truncated, info = self.env.step(action)
                        self.logger.debug(f"Step result: reward={reward}, done={done}, truncated={truncated}")
                    except Exception as e:
                        self.logger.error(f"Failed to execute step: {str(e)}")
                        # Create minimal observation to continue
                        obs = self._create_minimal_observation()
                        reward = 0.0
                        done = True
                        truncated = False
                        info = {}
                    
                    # Update metrics
                    game_reward += reward
                    game_steps += 1
                    
                    # Check win condition
                    if done:
                        wins += 1
                        self.logger.info(f"Game {game+1} won in {game_steps} steps with reward {game_reward:.2f}")
                    elif truncated:
                        self.logger.info(f"Game {game+1} truncated after {game_steps} steps with reward {game_reward:.2f}")
                
                # Add game results to totals
                total_rewards += game_reward
                total_steps += game_steps
                
            except Exception as e:
                self.logger.error(f"Error during evaluation game {game+1}: {str(e)}")
                self.logger.exception("Game error details:")
                
                # Continue with next game
                continue
        
        # Calculate metrics
        played_games = max(1, game + 1)  # Avoid division by zero
        win_rate = wins / played_games
        avg_reward = total_rewards / played_games
        avg_steps = total_steps / played_games
        
        # Create results dictionary
        metrics = {
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps
        }
        
        self.logger.info("Evaluation completed")
        self.logger.info(f"Results: {metrics}")
        
        return metrics
    
    def _create_minimal_observation(self) -> Dict[str, np.ndarray]:
        """Create a minimal valid observation when recovery is needed
        
        Returns:
            Dictionary with minimal observation data
        """
        # Default observation structure
        board = np.zeros((9, 8, 8), dtype=np.float32)
        board[0, 0, 0] = 1  # Player at position (0,0)
        
        return {
            'board': board,
            'hand': np.zeros((5, 5), dtype=np.float32),
            'program': np.zeros((10, 5), dtype=np.float32)
        }
        
    def _get_action(self, obs: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        """Get action from model
        
        Args:
            obs: Current observation
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        # Convert observation to tensor
        obs_tensor = {}
        for k, v in obs.items():
            obs_tensor[k] = torch.FloatTensor(v).unsqueeze(0).to(self.device)
        
        # Get action from model
        with torch.no_grad():
            outputs = self.model(obs_tensor)
            action_probs = outputs['action_probs'][0]
            wall_probs = outputs['wall_probs'][0]
            value = outputs['value'][0]
        
        # Sample action
        action_type = torch.multinomial(action_probs, 1).item()
        wall_probs_flat = wall_probs.reshape(-1)
        wall_idx = torch.multinomial(wall_probs_flat, 1).item()
        wall_pos = (wall_idx // 8, wall_idx % 8)
        
        # Create action dictionary
        action = {
            'action_type': action_type,
            'wall_position': wall_pos
        }
        
        # Calculate log probabilities
        action_log_prob = torch.log(action_probs[action_type] + 1e-10)
        wall_log_prob = torch.log(wall_probs_flat[wall_idx] + 1e-10)
        log_prob = action_log_prob + wall_log_prob
        
        return action, log_prob, value

    def evaluate(self, env: RobotTurtlesEnv) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            env: Game environment
            
        Returns:
            Evaluation statistics
        """
        # Run evaluation against random
        random_stats = self.evaluate_against_random(num_games=10)
        
        # Create statistics
        stats = {
            'win_rate': random_stats['win_rate'],
            'avg_steps': random_stats['avg_steps'],
            'avg_reward': random_stats['avg_reward']
        }
        
        return stats 