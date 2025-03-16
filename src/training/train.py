import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from ..models.network import RobotTurtlesNet
from ..environment.game_env import RobotTurtlesEnv, GameConfig
from .self_play import SelfPlay
from .data_generator import DataGenerator

def train(
    num_epochs=1000,
    batch_size=32,
    learning_rate=0.001,
    device='cuda',
    log_dir='logs'
):
    # 初始化模型和优化器
    model = RobotTurtlesNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir)
    
    # 初始化环境和训练组件
    config = GameConfig(enable_bug_card=True)
    env = RobotTurtlesEnv(config)
    self_play = SelfPlay(model, env)
    data_generator = DataGenerator(env)
    
    # 训练循环
    for epoch in tqdm(range(num_epochs)):
        # 收集训练数据
        game_data = []
        for _ in range(batch_size):
            if np.random.random() < 0.7:  # 70%使用自我对弈
                game_data.extend(self_play.play_game(temperature=1.0))
            else:  # 30%使用随机数据
                game_data.extend(data_generator.generate_random_game())
                
        # 准备批次数据
        states = []
        action_targets = []
        wall_targets = []
        rewards = []
        
        for data in game_data:
            states.append(data['state'])
            action_targets.append(data['action_probs'])
            wall_targets.append(data['wall_probs'])
            rewards.append(data['reward'])
            
        # 转换为张量
        states = {k: torch.FloatTensor(np.array([s[k] for s in states])).to(device) 
                 for k in states[0].keys()}
        action_targets = torch.FloatTensor(np.array(action_targets)).to(device)
        wall_targets = torch.FloatTensor(np.array(wall_targets)).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        
        # 前向传播
        outputs = model(states)
        action_probs = outputs['action_probs']
        wall_probs = outputs['wall_probs']
        
        # 计算损失
        action_loss = -torch.mean(action_targets * torch.log(action_probs + 1e-8) * rewards.unsqueeze(1))
        wall_loss = -torch.mean(wall_targets * torch.log(wall_probs + 1e-8) * rewards.unsqueeze(1).unsqueeze(2))
        total_loss = action_loss + wall_loss
        
        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 记录日志
        writer.add_scalar('Loss/total', total_loss.item(), epoch)
        writer.add_scalar('Loss/action', action_loss.item(), epoch)
        writer.add_scalar('Loss/wall', wall_loss.item(), epoch)
        writer.add_scalar('Reward/mean', rewards.mean().item(), epoch)
        
        # 定期保存模型
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
            }, f'checkpoints/model_epoch_{epoch+1}.pt')
            
    writer.close()
    return model 