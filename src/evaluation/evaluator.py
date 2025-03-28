import torch
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
        self.env = RobotTurtlesEnv(GameConfig())
        
        self.logger.info("Evaluator initialized successfully")
        
    def evaluate_against_random(self, num_games: int = 10) -> Dict[str, float]:
        """Evaluate the model against random opponent
        
        Args:
            num_games: Number of games to play
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Starting evaluation against random opponent for {num_games} games")
        
        wins = 0
        total_rewards = 0
        total_steps = 0
        
        for game in range(num_games):
            self.logger.info(f"Playing game {game+1}/{num_games}")
            
            obs, _ = self.env.reset()
            done = False
            truncated = False
            game_reward = 0
            game_steps = 0
            
            while not (done or truncated):
                # Get model action
                action, _, _ = self._get_action(obs)
                
                # Execute action
                obs, reward, done, truncated, info = self.env.step(action)
                
                game_reward += reward
                game_steps += 1
                
                if done:
                    wins += 1
                    self.logger.info(f"Game {game+1} won in {game_steps} steps")
                elif truncated:
                    self.logger.info(f"Game {game+1} truncated after {game_steps} steps")
            
            total_rewards += game_reward
            total_steps += game_steps
        
        # Calculate metrics
        win_rate = wins / num_games
        avg_reward = total_rewards / num_games
        avg_steps = total_steps / num_games
        
        metrics = {
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps
        }
        
        self.logger.info("Evaluation completed")
        self.logger.info(f"Results: {metrics}")
        
        return metrics
    
    def evaluate_self_play(
        self, 
        num_games: int = 50,
        env_config: Optional[GameConfig] = None
    ) -> Dict[str, float]:
        """
        自我对弈评估
        
        Args:
            num_games: 评估游戏局数
            env_config: 环境配置
            
        Returns:
            评估统计信息
        """
        config = env_config or GameConfig(num_players=2)
        env = RobotTurtlesEnv(config)
        
        stats = {
            'player1_wins': 0,
            'avg_game_length': 0,
            'avg_total_reward': 0,
            'total_games': num_games
        }
        
        for _ in range(num_games):
            obs = env.reset()[0]
            done = False
            steps = 0
            total_reward = 0
            
            while not done:
                action = self._get_action(obs)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                steps += 1
                
            if info.get('winner', -1) == 0:
                stats['player1_wins'] += 1
            stats['avg_game_length'] += steps
            stats['avg_total_reward'] += total_reward
            
        # 计算平均值
        stats['player1_win_rate'] = stats['player1_wins'] / num_games
        stats['avg_game_length'] /= num_games
        stats['avg_total_reward'] /= num_games
        
        return stats
    
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
        wall_idx = torch.multinomial(wall_probs.reshape(-1), 1).item()
        wall_pos = (wall_idx // 8, wall_idx % 8)
        
        # Create action dictionary
        action = {
            'action_type': action_type,
            'wall_position': wall_pos
        }
        
        # Calculate log probabilities
        action_log_prob = torch.log(action_probs[action_type] + 1e-10)
        wall_log_prob = torch.log(wall_probs.reshape(-1)[wall_idx] + 1e-10)
        log_prob = action_log_prob + wall_log_prob
        
        return action, log_prob, value

    def evaluate(self, env: RobotTurtlesEnv) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            env: 游戏环境
            
        Returns:
            评估统计信息
        """
        # 保存环境配置
        env_config = env.config
        
        # 运行两种评估
        random_stats = self.evaluate_against_random(num_games=10)
        self_play_stats = self.evaluate_self_play(num_games=5, env_config=env_config)
        
        # 合并统计信息
        stats = {
            'vs_random_win_rate': random_stats['win_rate'],
            'vs_random_avg_steps': random_stats['avg_steps'],
            'vs_random_avg_reward': random_stats['avg_reward'],
            'self_play_win_rate': self_play_stats['player1_win_rate'],
            'self_play_avg_steps': self_play_stats['avg_game_length'],
            'self_play_avg_reward': self_play_stats['avg_total_reward']
        }
        
        return stats 