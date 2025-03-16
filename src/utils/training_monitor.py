import time
from typing import Dict, Any
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class TrainingMonitor:
    def __init__(self, log_dir: str, experiment_name: str):
        """
        初始化训练监控器
        Args:
            log_dir: tensorboard日志目录
            experiment_name: 实验名称
        """
        self.writer = SummaryWriter(f"{log_dir}/{experiment_name}")
        self.start_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0
        
    def log_step(self, info: Dict[str, Any], step: int):
        """
        记录每个训练步骤的信息
        Args:
            info: 包含训练信息的字典
            step: 当前步数
        """
        # 记录损失
        if "loss" in info:
            self.writer.add_scalar("train/loss", info["loss"], step)
        
        # 记录策略损失
        if "policy_loss" in info:
            self.writer.add_scalar("train/policy_loss", info["policy_loss"], step)
            
        # 记录值函数损失
        if "value_loss" in info:
            self.writer.add_scalar("train/value_loss", info["value_loss"], step)
            
        # 记录熵损失
        if "entropy_loss" in info:
            self.writer.add_scalar("train/entropy_loss", info["entropy_loss"], step)
            
        # 记录学习率
        if "learning_rate" in info:
            self.writer.add_scalar("train/learning_rate", info["learning_rate"], step)
            
        # 记录奖励
        if "reward" in info:
            self.episode_rewards.append(info["reward"])
            
        # 记录回合长度
        if "episode_length" in info:
            self.episode_lengths.append(info["episode_length"])
            
        self.total_steps = step
        
    def log_episode(self, episode: int):
        """
        记录每个回合的统计信息
        Args:
            episode: 当前回合数
        """
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards[-100:])
            self.writer.add_scalar("train/average_reward", avg_reward, episode)
            self.writer.add_scalar("train/episode_reward", self.episode_rewards[-1], episode)
            
        if self.episode_lengths:
            avg_length = np.mean(self.episode_lengths[-100:])
            self.writer.add_scalar("train/average_length", avg_length, episode)
            self.writer.add_scalar("train/episode_length", self.episode_lengths[-1], episode)
            
        # 记录训练时间
        elapsed_time = time.time() - self.start_time
        self.writer.add_scalar("train/elapsed_time", elapsed_time, episode)
        
    def close(self):
        """关闭 SummaryWriter"""
        self.writer.close()
        
    def get_stats(self) -> Dict[str, float]:
        """获取训练统计信息"""
        stats = {
            "total_steps": self.total_steps,
            "elapsed_time": time.time() - self.start_time
        }
        
        if self.episode_rewards:
            stats.update({
                "average_reward": np.mean(self.episode_rewards[-100:]),
                "latest_reward": self.episode_rewards[-1]
            })
            
        if self.episode_lengths:
            stats.update({
                "average_length": np.mean(self.episode_lengths[-100:]),
                "latest_length": self.episode_lengths[-1]
            })
            
        return stats
