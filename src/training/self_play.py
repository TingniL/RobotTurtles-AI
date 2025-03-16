import torch
import numpy as np
from ..environment.game_env import RobotTurtlesEnv

class SelfPlay:
    """自我对弈训练"""
    
    def __init__(self, model):
        self.env = RobotTurtlesEnv()
        self.model = model
    
    def play_game(self, temperature=1.0):
        """进行一局自我对弈"""
        obs = self.env.reset()
        done = False
        game_data = []
        
        while not done:
            # 使用模型预测动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_probs = self.model(state_tensor)
                
            # 根据温度参数采样动作
            action_probs = action_probs.numpy()
            action_probs = np.power(action_probs, 1/temperature)
            action_probs = action_probs / np.sum(action_probs)
            action = np.random.choice(len(action_probs), p=action_probs)
            
            next_obs, reward, done, _, info = self.env.step(action)
            
            game_data.append({
                'state': obs,
                'action_probs': action_probs,
                'reward': reward
            })
            
            obs = next_obs
            
        return game_data 