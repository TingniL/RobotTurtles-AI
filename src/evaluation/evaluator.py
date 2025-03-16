import torch
import numpy as np
from typing import List, Dict
from ..environment.game_env import RobotTurtlesEnv, GameConfig
from ..models.network import RobotTurtlesNet

class Evaluator:
    """模型评估器"""
    
    def __init__(self, model: RobotTurtlesNet, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def evaluate_against_random(self, num_games: int = 100) -> Dict:
        """对抗随机玩家的评估"""
        config = GameConfig(num_players=2)
        env = RobotTurtlesEnv(config)
        
        stats = {
            'wins': 0,
            'avg_steps': 0,
            'avg_reward': 0
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
                    action = {
                        'action_type': env.action_space['action_type'].sample(),
                        'wall_position': (
                            np.random.randint(8),
                            np.random.randint(8)
                        )
                    }
                
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
                steps += 1
                
            if total_reward > 0:  # AI获胜
                stats['wins'] += 1
            stats['avg_steps'] += steps
            stats['avg_reward'] += total_reward
            
        stats['win_rate'] = stats['wins'] / num_games
        stats['avg_steps'] /= num_games
        stats['avg_reward'] /= num_games
        
        return stats
    
    def evaluate_self_play(self, num_games: int = 50) -> Dict:
        """自我对弈评估"""
        config = GameConfig(num_players=2)
        env = RobotTurtlesEnv(config)
        
        stats = {
            'player1_wins': 0,
            'avg_game_length': 0,
            'avg_total_reward': 0
        }
        
        for _ in range(num_games):
            obs = env.reset()[0]
            done = False
            steps = 0
            total_reward = 0
            
            while not done:
                action = self._get_action(obs)
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
                steps += 1
                
            if env.players[0].score > 0:
                stats['player1_wins'] += 1
            stats['avg_game_length'] += steps
            stats['avg_total_reward'] += total_reward
            
        stats['player1_win_rate'] = stats['player1_wins'] / num_games
        stats['avg_game_length'] /= num_games
        stats['avg_total_reward'] /= num_games
        
        return stats
    
    def _get_action(self, obs: Dict) -> Dict:
        """获取模型预测的动作"""
        with torch.no_grad():
            # 转换观察为tensor
            state = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) 
                    for k, v in obs.items()}
            
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