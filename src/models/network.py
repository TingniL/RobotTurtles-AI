import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class RobotTurtlesNet(nn.Module):
    """机器海龟神经网络模型"""
    
    def __init__(self, input_channels: int = 9):
        """
        Args:
            input_channels: 输入通道数
                - 2个玩家位置
                - 2种墙
                - 4个方向
                - 1个BugCard状态
        """
        super().__init__()
        
        # 棋盘特征提取
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 手牌特征提取 (5张牌 x 5种类型)
        self.hand_fc = nn.Linear(25, 64)
        
        # 程序特征提取 (20步 x 5种卡牌)
        self.program_fc = nn.Linear(100, 64)
        
        # 合并特征后的全连接层
        self.fc1 = nn.Linear(64 * 8 * 8 + 64 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # 动作类型头 (8个动作类型)
        self.action_type = nn.Linear(256, 8)
        
        # 墙放置位置头 (8x8棋盘位置)
        self.wall_position = nn.Linear(256, 64)
        
        # 价值头
        self.value = nn.Linear(256, 1)
        
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 包含board, hand, program的字典
                - board: [batch_size, channels, 8, 8]
                - hand: [batch_size, 5, 5]
                - program: [batch_size, 20, 5]
                
        Returns:
            包含action_probs, wall_probs, value的字典
        """
        # 解包输入
        x_board = x['board']
        x_hand = x['hand']
        x_program = x['program']
        
        # 确保输入维度正确
        batch_size = x_board.size(0)
        x_hand = x_hand.reshape(batch_size, -1)
        x_program = x_program.reshape(batch_size, -1)
        
        # 特征提取
        x_board = F.relu(self.conv1(x_board))
        x_board = F.relu(self.conv2(x_board))
        x_board = F.relu(self.conv3(x_board))
        x_board = x_board.reshape(batch_size, -1)
        
        x_hand = F.relu(self.hand_fc(x_hand))
        x_program = F.relu(self.program_fc(x_program))
        
        # 合并特征
        x = torch.cat([x_board, x_hand, x_program], dim=1)
        
        # 共享特征提取
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 输出动作概率、墙放置概率和价值
        action_probs = F.softmax(self.action_type(x), dim=1)
        # 修改这一行，先展平后再用softmax
        wall_logits = self.wall_position(x).reshape(batch_size, -1)  # 展平为 (batch_size, 64)
        wall_probs = F.softmax(wall_logits, dim=1).reshape(batch_size, 8, 8)  # 重塑回 (batch_size, 8, 8)
        value = self.value(x)
        
        return {
            'action_probs': action_probs,
            'wall_probs': wall_probs,
            'value': value
        }