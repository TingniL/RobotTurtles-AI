import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from ..models.network import RobotTurtlesNet
from ..environment.game_env import RobotTurtlesEnv

class PPOTrainer:
    """PPO算法训练器"""
    
    def __init__(
        self,
        model: RobotTurtlesNet,
        env: RobotTurtlesEnv,
        device: str = 'cuda',
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        self.model = model
        self.env = env
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def collect_trajectories(
        self, 
        num_steps: int = 2048
    ) -> List[Dict]:
        """收集轨迹数据"""
        trajectories = []
        obs = self.env.reset()[0]
        
        for _ in range(num_steps):
            # 获取动作
            with torch.no_grad():
                state = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) 
                        for k, v in obs.items()}
                outputs = self.model(state)
                
                action_probs = outputs['action_probs'][0]
                wall_probs = outputs['wall_probs'][0]
                value = outputs.get('value', torch.zeros(1))[0]
                
                action_dist = torch.distributions.Categorical(action_probs)
                action_type = action_dist.sample().item()
                
                if action_type in [4, 5]:
                    wall_dist = torch.distributions.Categorical(wall_probs.view(-1))
                    wall_idx = wall_dist.sample().item()
                    wall_pos = (wall_idx // 8, wall_idx % 8)
                else:
                    wall_pos = (0, 0)
                    
            # 执行动作
            action = {
                'action_type': action_type,
                'wall_position': wall_pos
            }
            next_obs, reward, done, _, _ = self.env.step(action)
            
            # 保存轨迹
            trajectories.append({
                'state': obs,
                'action': action,
                'reward': reward,
                'done': done,
                'value': value.cpu().numpy(),
                'action_log_prob': action_dist.log_prob(
                    torch.tensor(action_type)
                ).cpu().numpy(),
                'wall_log_prob': (
                    torch.distributions.Categorical(
                        wall_probs.view(-1)
                    ).log_prob(torch.tensor(wall_idx)).cpu().numpy()
                    if action_type in [4, 5] else 0
                )
            })
            
            if done:
                obs = self.env.reset()[0]
            else:
                obs = next_obs
                
        return trajectories
    
    def update(self, trajectories: List[Dict]) -> Dict:
        """更新策略"""
        # 计算优势值
        advantages = self._compute_advantages(trajectories)
        
        # 准备数据
        states = []
        actions = []
        wall_positions = []
        old_action_log_probs = []
        old_wall_log_probs = []
        returns = []
        
        for traj, adv in zip(trajectories, advantages):
            states.append(traj['state'])
            actions.append(traj['action']['action_type'])
            wall_positions.append(traj['action']['wall_position'])
            old_action_log_probs.append(traj['action_log_prob'])
            old_wall_log_probs.append(traj['wall_log_prob'])
            returns.append(adv + traj['value'])
            
        # 转换为tensor
        states = {k: torch.FloatTensor(np.array([s[k] for s in states])).to(self.device) 
                 for k in states[0].keys()}
        actions = torch.LongTensor(actions).to(self.device)
        wall_positions = torch.LongTensor(wall_positions).to(self.device)
        old_action_log_probs = torch.FloatTensor(old_action_log_probs).to(self.device)
        old_wall_log_probs = torch.FloatTensor(old_wall_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 多次更新
        for _ in range(10):
            # 获取新的动作概率和值
            outputs = self.model(states)
            action_probs = outputs['action_probs']
            wall_probs = outputs['wall_probs']
            values = outputs.get('value', torch.zeros_like(returns))
            
            # 计算新的log概率
            action_dist = torch.distributions.Categorical(action_probs)
            new_action_log_probs = action_dist.log_prob(actions)
            
            wall_dist = torch.distributions.Categorical(wall_probs.view(wall_probs.size(0), -1))
            wall_indices = wall_positions[:, 0] * 8 + wall_positions[:, 1]
            new_wall_log_probs = wall_dist.log_prob(wall_indices)
            
            # 计算比率
            action_ratio = torch.exp(new_action_log_probs - old_action_log_probs)
            wall_ratio = torch.exp(new_wall_log_probs - old_wall_log_probs)
            
            # 计算策略损失
            action_loss1 = action_ratio * advantages
            action_loss2 = torch.clamp(action_ratio, 1-self.epsilon, 1+self.epsilon) * advantages
            action_loss = -torch.min(action_loss1, action_loss2).mean()
            
            wall_loss1 = wall_ratio * advantages
            wall_loss2 = torch.clamp(wall_ratio, 1-self.epsilon, 1+self.epsilon) * advantages
            wall_loss = -torch.min(wall_loss1, wall_loss2).mean()
            
            # 计算值损失
            value_loss = 0.5 * (returns - values).pow(2).mean()
            
            # 计算熵损失
            entropy_loss = -(
                action_dist.entropy().mean() + 
                wall_dist.entropy().mean()
            )
            
            # 总损失
            total_loss = (
                action_loss + 
                wall_loss + 
                self.value_coef * value_loss + 
                self.entropy_coef * entropy_loss
            )
            
            # 更新模型
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
        return {
            'action_loss': action_loss.item(),
            'wall_loss': wall_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
        
    def _compute_advantages(
        self, 
        trajectories: List[Dict]
    ) -> np.ndarray:
        """计算优势值"""
        rewards = np.array([t['reward'] for t in trajectories])
        values = np.array([t['value'] for t in trajectories])
        dones = np.array([t['done'] for t in trajectories])
        
        # 计算TD误差
        deltas = rewards + self.gamma * np.append(values[1:], 0) * (1 - dones) - values
        
        # 计算GAE
        advantages = np.zeros_like(deltas)
        advantage = 0
        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + self.gamma * 0.95 * advantage * (1 - dones[t])
            advantages[t] = advantage
            
        # 标准化优势值
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages 