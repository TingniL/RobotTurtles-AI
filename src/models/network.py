import torch
import torch.nn as nn
import torch.nn.functional as F

class RobotTurtlesNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 棋盘特征提取
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 手牌特征提取
        self.hand_fc = nn.Linear(25, 64)  # 5x5 的手牌
        
        # 程序特征提取
        self.program_fc = nn.Linear(100, 64)  # 20x5 的程序
        
        # 合并特征后的全连接层
        self.fc1 = nn.Linear(64 * 8 * 8 + 64 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # 策略头
        self.policy = nn.Linear(256, 8)  # 8个动作
        
        # 价值头
        self.value = nn.Linear(256, 1)
        
    def forward(self, x):
        # 解包输入
        x_board = x['board']  # [batch_size, channels, 8, 8]
        x_hand = x['hand']    # [batch_size, 5, 5]
        x_program = x['program']  # [batch_size, 20, 5]
        
        # 确保输入维度正确
        batch_size = x_board.size(0)
        x_hand = x_hand.reshape(batch_size, -1)  # 展平为 [batch_size, 25]
        x_program = x_program.reshape(batch_size, -1)  # 展平为 [batch_size, 100]
        
        # 处理棋盘
        x_board = F.relu(self.conv1(x_board))
        x_board = F.relu(self.conv2(x_board))
        x_board = F.relu(self.conv3(x_board))
        x_board = x_board.reshape(batch_size, -1)  # 使用 reshape 替代 view
        
        # 处理手牌
        x_hand = F.relu(self.hand_fc(x_hand))
        
        # 处理程序
        x_program = F.relu(self.program_fc(x_program))
        
        # 合并特征
        x = torch.cat([x_board, x_hand, x_program], dim=1)
        
        # 共享特征提取
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 输出策略和价值
        policy = F.softmax(self.policy(x), dim=1)
        value = self.value(x)
        
        return policy, value