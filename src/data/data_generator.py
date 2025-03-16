import numpy as np
from ..environment.game_env import RobotTurtlesEnv

class DataGenerator:
    """生成训练数据"""
    
    def __init__(self):
        self.env = RobotTurtlesEnv()
    
    def generate_random_game(self):
        """生成一局随机游戏数据"""
        obs = self.env.reset()
        done = False
        game_data = []
        
        while not done:
            # 随机选择动作
            action = self.env.action_space.sample()
            next_obs, reward, done, _, info = self.env.step(action)
            
            game_data.append({
                'state': obs,
                'action': action,
                'reward': reward,
                'next_state': next_obs,
                'done': done
            })
            
            obs = next_obs
            
        return game_data 