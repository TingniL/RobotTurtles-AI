import torch
import torch.nn as nn
import torch.optim as optim
from .self_play import SelfPlay
from ..data.data_generator import DataGenerator

class Trainer:
    """模型训练器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters())
        self.self_play = SelfPlay(model)
        self.data_generator = DataGenerator()
    
    def train_step(self, states, action_probs, rewards):
        """执行一步训练"""
        states = torch.FloatTensor(states).to(self.device)
        action_probs = torch.FloatTensor(action_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        self.optimizer.zero_grad()
        
        # 前向传播
        pred_probs = self.model(states)
        
        # 计算损失
        policy_loss = -torch.mean(action_probs * torch.log(pred_probs) * rewards)
        
        # 反向传播
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item() 