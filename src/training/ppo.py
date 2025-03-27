import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import logging
import os
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional
from multiprocessing import Pool, set_start_method
from collections import defaultdict

from src.models.network import RobotTurtlesNet

# 设置多进程启动方法为 spawn
try:
    set_start_method('spawn')
except RuntimeError:
    pass  # 如果已经设置过则忽略

class PPODataset(Dataset):
    """PPO训练数据集，支持字典输入"""
    
    def __init__(self, observations, actions, old_log_probs, returns, advantages):
        """初始化数据集
        
        Args:
            observations: 观察字典，包含多个张量
            actions: 动作字典，包含动作类型和墙放置位置
            old_log_probs: 旧的动作对数概率
            returns: 动作回报
            advantages: 动作优势值
        """
        self.observations = observations
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.returns = returns
        self.advantages = advantages
        self.length = len(old_log_probs)  # 使用其中一个张量的长度作为数据集长度
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """获取数据项"""
        # 从字典中取出对应的张量
        obs = {k: v[idx] for k, v in self.observations.items()}
        action = {k: v[idx] for k, v in self.actions.items()}
        
        # 返回观察、动作、旧对数概率、回报和优势值
        return (
            obs,
            action,
            self.old_log_probs[idx],
            self.returns[idx],
            self.advantages[idx]
        )

class PPOTrainer:
    """PPO算法训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        env: Any,
        device: torch.device,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs: int = 10,
        num_processes: int = 8,
        parallel_envs: int = 16,
        batch_size: int = 64,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True
    ):
        """
        Args:
            model: 策略网络
            env: 游戏环境
            device: 计算设备
            learning_rate: 学习率
            clip_ratio: PPO裁剪比例
            value_coef: 价值损失系数
            entropy_coef: 熵损失系数
            epochs: 每次更新的训练轮数
            num_processes: 并行进程数
            parallel_envs: 并行环境数
            batch_size: 批次大小
            num_workers: 数据加载器工作线程数
            prefetch_factor: 数据加载器预取因子
            pin_memory: 是否固定内存
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化 PPOTrainer...")
        
        self.model = model
        self.env = env
        self.device = device
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.num_processes = num_processes
        self.parallel_envs = parallel_envs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        
        # 使用 FusedAdam 优化器提高性能
        try:
            from apex.optimizers import FusedAdam
            self.optimizer = FusedAdam(model.parameters(), lr=learning_rate)
            self.logger.info("使用 FusedAdam 优化器")
        except ImportError:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            self.logger.info("使用标准 Adam 优化器")
        
        # 创建并行环境
        self.logger.info(f"创建 {self.parallel_envs} 个并行环境...")
        self.envs = [deepcopy(env) for _ in range(self.parallel_envs)]
        
        # 设置自动混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
        self.logger.info("启用自动混合精度训练")
        
        # 创建数据加载器
        self.data_loader = None
        self.logger.info("PPOTrainer 初始化完成")

    def collect_trajectories(self, num_steps: int) -> List[Dict]:
        """简化的数据收集，使用单进程"""
        self.logger.info(f"开始单进程收集 {num_steps} 步轨迹数据...")
        trajectories = []
        env = deepcopy(self.env)
        obs, _ = env.reset()
        
        # 限制最大收集时间
        max_steps = min(num_steps, 100)  # 限制最大步数
        
        for step in range(max_steps):
            action, log_prob, value = self.get_action_and_value(obs)
            next_obs, reward, done, _, info = env.step(action)
            
            trajectories.append({
                'observation': obs,
                'action': action,
                'reward': reward,
                'next_observation': next_obs,
                'done': done,
                'info': info,
                'log_prob': log_prob,
                'value': value
            })
            
            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs
        
        self.logger.info(f"成功收集了 {len(trajectories)} 条轨迹")
        return trajectories

    def collect_trajectories_single_process(self, num_steps: int) -> List[Dict]:
        """使用单进程收集轨迹数据"""
        self.logger.info(f"开始单进程收集 {num_steps} 步轨迹数据...")
        
        # 创建环境实例
        env = deepcopy(self.env)
        
        # 收集数据
        trajectories = []
        obs, _ = env.reset()
        
        for step in range(num_steps):
            try:
                # 获取动作
                action, log_prob, value = self.get_action_and_value(obs)
                
                # 执行动作
                next_obs, reward, done, _, info = env.step(action)
                
                # 保存轨迹
                trajectories.append({
                    'observation': obs,
                    'action': action,
                    'reward': reward,
                    'next_observation': next_obs,
                    'done': done,
                    'info': info,
                    'log_prob': log_prob,
                    'value': value
                })
                
                if done:
                    obs, _ = env.reset()
                else:
                    obs = next_obs
                    
            except Exception as e:
                self.logger.error(f"收集数据时发生错误: {str(e)}")
                self.logger.exception("详细错误信息:")
                
        self.logger.info(f"成功收集了 {len(trajectories)} 条轨迹")
        
        try:
            env.close()
        except:
            pass
            
        return trajectories

    def update(self, trajectories: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """更新策略"""
        self.logger.info("开始更新策略...")
        
        # 准备训练数据
        observations, actions, old_log_probs, returns, advantages = self._prepare_training_data(trajectories)
        
        # 创建自定义数据集
        dataset = PPODataset(
            observations,
            actions,
            old_log_probs,
            returns,
            advantages
        )
        
        # 创建数据加载器
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory
        )
        self.logger.info(f"创建数据加载器，批次大小: {self.batch_size}")
        
        metrics = defaultdict(float)
        
        for epoch in range(self.epochs):
            self.logger.info(f"PPO更新轮次 {epoch+1}/{self.epochs}")
            for batch_idx, batch in enumerate(self.data_loader):
                # 转移数据到设备
                obs_batch, action_batch, old_log_prob_batch, return_batch, advantage_batch = batch
                
                # 使用自动混合精度
                with torch.cuda.amp.autocast():
                    # 前向传播
                    outputs = self.model(obs_batch)
                    action_probs = outputs['action_probs']
                    wall_probs = outputs['wall_probs']
                    value = outputs['value'].squeeze(-1)
                    
                    # 计算新的动作概率
                    action_types = action_batch['action_type']
                    wall_positions = action_batch['wall_position']
                    wall_indices = wall_positions[:, 0] * 8 + wall_positions[:, 1]
                    
                    new_action_log_probs = torch.log(action_probs.gather(1, action_types.unsqueeze(-1)).squeeze(-1) + 1e-10)
                    new_wall_log_probs = torch.log(wall_probs.reshape(wall_probs.shape[0], -1).gather(1, wall_indices.unsqueeze(-1)).squeeze(-1) + 1e-10)
                    new_log_probs = new_action_log_probs + new_wall_log_probs
                    
                    # 计算损失
                    ratio = torch.exp(new_log_probs - old_log_prob_batch)
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    
                    policy_loss = -torch.min(
                        ratio * advantage_batch,
                        clipped_ratio * advantage_batch
                    ).mean()
                    
                    value_loss = F.mse_loss(value, return_batch)
                    entropy_loss = -(new_action_log_probs.mean() + new_wall_log_probs.mean())
                    
                    total_loss = (
                        policy_loss +
                        self.value_coef * value_loss +
                        self.entropy_coef * entropy_loss
                    )
                
                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # 更新指标
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy_loss'] += entropy_loss.item()
                metrics['total_loss'] += total_loss.item()
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"  批次 {batch_idx}/{len(self.data_loader)}, "
                                   f"总损失: {total_loss.item():.4f}, "
                                   f"策略损失: {policy_loss.item():.4f}, "
                                   f"价值损失: {value_loss.item():.4f}, "
                                   f"熵损失: {entropy_loss.item():.4f}")
        
        # 计算平均值
        num_batches = len(self.data_loader) * self.epochs
        for k in metrics:
            metrics[k] /= num_batches
        
        self.logger.info("策略更新完成")
        self.logger.info(f"平均指标: {metrics}")
        return dict(metrics)

    def _compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """计算优势值
        
        Args:
            rewards: 奖励张量
            values: 价值张量
            dones: 完成标志张量
            
        Returns:
            优势值张量
        """
        advantages = torch.zeros_like(rewards)
        last_value = 0
        last_advantage = 0
        gamma = 0.99
        lambda_ = 0.95
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_advantage = delta
            else:
                delta = rewards[t] + gamma * last_value - values[t]
                last_advantage = delta + gamma * lambda_ * last_advantage
            advantages[t] = last_advantage
            last_value = values[t]
            
        return advantages

    def _compute_returns(self, trajectories: List[Dict[str, Any]]) -> np.ndarray:
        """计算回报"""
        returns = np.zeros(len(trajectories))
        last_return = 0
        gamma = 0.99
        
        for t in reversed(range(len(trajectories))):
            if trajectories[t]['done']:
                last_return = trajectories[t]['reward']
            else:
                last_return = trajectories[t]['reward'] + gamma * last_return
            returns[t] = last_return
            
        return returns

    def get_action_and_value(self, obs: Dict[str, torch.Tensor]) -> Tuple[Action, torch.Tensor, torch.Tensor]:
        """获取动作和价值
        
        Args:
            obs: 观察字典
            
        Returns:
            action: 动作字典
            log_prob: 动作的对数概率
            value: 状态价值
        """
        # 转换观察为张量
        obs_tensor = {}
        for k, v in obs.items():
            # 检查是否已经是张量
            if not isinstance(v, torch.Tensor):
                obs_tensor[k] = torch.FloatTensor(v).unsqueeze(0)
            else:
                obs_tensor[k] = v.unsqueeze(0)
            
        # 转移到设备
        if self.device.type != 'cpu':
            for k in obs_tensor:
                obs_tensor[k] = obs_tensor[k].to(self.device, non_blocking=True)
        
        # 获取动作概率和价值
        with torch.no_grad():
            outputs = self.model(obs_tensor)
            action_probs = outputs['action_probs'][0]
            wall_probs = outputs['wall_probs'][0]
            value = outputs['value'][0]
        
        # 采样动作
        action_type = torch.multinomial(action_probs, 1).item()
        wall_idx = torch.multinomial(wall_probs.reshape(-1), 1).item()
        wall_pos = (wall_idx // 8, wall_idx % 8)  # 直接计算行列索引
        
        # 构造action字典
        action = {
            'action_type': action_type,
            'wall_position': wall_pos
        }
        
        # 计算动作的对数概率
        action_log_prob = torch.log(action_probs[action_type] + 1e-10)
        wall_log_prob = torch.log(wall_probs.reshape(-1)[wall_idx] + 1e-10)
        log_prob = action_log_prob + wall_log_prob
        
        return action, log_prob, value

    def _prepare_training_data(self, trajectories: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
        """准备训练数据
        
        Args:
            trajectories: 轨迹列表
            
        Returns:
            训练数据元组
        """
        # 提取数据
        observations = []
        action_types = []
        wall_positions = []
        old_log_probs = []
        rewards = []
        values = []
        dones = []
        
        for t in trajectories:
            observations.append(t['observation'])
            
            # 确保动作是字典形式，处理不同数据类型
            action = t['action']
            if isinstance(action, dict):
                action_type = action['action_type']
                wall_position = action['wall_position']
            else:
                # 如果不是字典，尝试转换
                action_type = action
                wall_position = (0, 0)
                
            # 确保正确的数据类型
            if isinstance(action_type, torch.Tensor):
                action_types.append(action_type)
            else:
                action_types.append(torch.tensor(action_type, dtype=torch.int64))
                
            if isinstance(wall_position, torch.Tensor):
                wall_positions.append(wall_position)
            else:
                wall_positions.append(torch.tensor(wall_position, dtype=torch.int64))
                
            old_log_probs.append(t['log_prob'])
            rewards.append(t['reward'])
            values.append(t['value'])
            dones.append(t['done'])
        
        # 将NumPy观察转换为PyTorch张量
        processed_obs = {}
        for k in observations[0].keys():
            # 首先转换每个观察中的NumPy数组为PyTorch张量
            tensors = []
            for obs in observations:
                if isinstance(obs[k], np.ndarray):
                    tensors.append(torch.from_numpy(obs[k]).float())
                elif isinstance(obs[k], torch.Tensor):
                    tensors.append(obs[k].float())
                else:
                    tensors.append(torch.tensor(obs[k], dtype=torch.float32))
            # 然后堆叠这些张量
            processed_obs[k] = torch.stack(tensors).to(self.device)
        
        # 使用正确的方法处理tensors
        action_types = torch.stack([t if t.dim() > 0 else t.unsqueeze(0) for t in action_types]).to(self.device)
        wall_positions = torch.stack([t if t.dim() > 0 else t.unsqueeze(0) for t in wall_positions]).to(self.device)
        
        # 处理其他张量
        old_log_probs = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32) for t in old_log_probs]).to(self.device)
        rewards = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32) for t in rewards]).to(self.device)
        values = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32) for t in values]).to(self.device)
        dones = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.bool) for t in dones]).to(self.device)
        
        # 计算优势和回报
        advantages = self._compute_advantages(rewards, values, dones)
        returns = advantages + values
        
        # 构造动作字典
        actions = {
            'action_type': action_types,
            'wall_position': wall_positions
        }
        
        return processed_obs, actions, old_log_probs, returns, advantages