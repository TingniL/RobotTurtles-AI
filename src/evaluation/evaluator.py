import torch
import numpy as np
from typing import Dict, List, Any, Optional
from ..environment.game_env import RobotTurtlesEnv, GameConfig, Action
from ..models.network import RobotTurtlesNet

class Evaluator:
    """模型评估器"""
    
    def __init__(self, model: RobotTurtlesNet, device: str = 'cuda'):
        """
        初始化评估器
        
        Args:
            model: 策略网络
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.eval()
        
    def evaluate_against_random(
        self, 
        num_games: int = 100,
        env_config: Optional[GameConfig] = None
    ) -> Dict[str, float]:
        """
        对抗随机玩家的评估
        
        Args:
            num_games: 评估游戏局数
            env_config: 环境配置
            
        Returns:
            评估统计信息
        """
        config = env_config or GameConfig(num_players=2)
        env = RobotTurtlesEnv(config)
        
        stats = {
            'wins': 0,
            'avg_steps': 0,
            'avg_reward': 0,
            'total_games': num_games
        }
        
        for _ in range(num_games):
            obs = env.reset()[0]
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                if env.current_player == 0:  # AI玩家
                    action = self._get_action(obs)
                else:  # 随机玩家
                    valid_actions = env.get_valid_actions()
                    action = valid_actions[np.random.randint(len(valid_actions))]
                
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                steps += 1
                
            if info.get('winner', -1) == 0:  # AI获胜
                stats['wins'] += 1
            stats['avg_steps'] += steps
            stats['avg_reward'] += total_reward
            
        # 计算平均值
        stats['win_rate'] = stats['wins'] / num_games
        stats['avg_steps'] /= num_games
        stats['avg_reward'] /= num_games
        
        return stats
    
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
    
    def _get_action(self, obs: Dict[str, np.ndarray]) -> Action:
        """
        获取模型预测的动作
        
        Args:
            obs: 观察空间
            
        Returns:
            预测的动作
        """
        with torch.no_grad():
            # 转换观察为tensor
            state = {
                k: torch.FloatTensor(v).unsqueeze(0).to(self.device) 
                for k, v in obs.items()
            }
            
            # 获取模型输出
            outputs = self.model(state)
            action_probs = outputs['action_probs'][0].cpu().numpy()
            wall_probs = outputs['wall_probs'][0].cpu().numpy()
            
            # 选择动作
            action_type = np.argmax(action_probs)
            if action_type in [4, 5]:  # 放墙动作
                wall_pos = np.unravel_index(
                    np.argmax(wall_probs), 
                    wall_probs.shape
                )
            else:
                wall_pos = (0, 0)  # 默认值
                
            return {
                'action_type': action_type,
                'wall_position': wall_pos
            }

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
        random_stats = self.evaluate_against_random(num_games=10, env_config=env_config)
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