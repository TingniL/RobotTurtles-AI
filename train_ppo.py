import torch
import os
from datetime import datetime
import yaml
import argparse

# 修改导入路径
from src.models.network import RobotTurtlesNet
from src.environment.game_env import RobotTurtlesEnv, GameConfig
from src.training.ppo import PPOTrainer
from src.evaluation.evaluator import Evaluator
from src.visualization.visualizer import GameVisualizer
from torch.utils.tensorboard import SummaryWriter
from src.utils.checkpointing import CheckpointManager
from src.utils.training_monitor import TrainingMonitor

def main():
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'outputs/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{output_dir}/visualizations', exist_ok=True)
    
    # 初始化组件
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GameConfig(enable_bug_card=True)
    env = RobotTurtlesEnv(config)
    model = RobotTurtlesNet().to(device)
    trainer = PPOTrainer(model, env, device)
    evaluator = Evaluator(model, device)
    visualizer = GameVisualizer()
    writer = SummaryWriter(f'{output_dir}/logs')
    
    # 加载配置
    with open('config/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 检查是否有检查点
    checkpoint_path = args.resume_from if args.resume_from else None
    start_epoch = 0
    metrics = {}  # 初始化 metrics
    random_stats = {}  # 初始化 random_stats
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    monitor = TrainingMonitor(
        log_dir=f"{output_dir}/logs",  # 修改日志目录
        experiment_name="ppo_training"
    )
    checkpoint_manager = CheckpointManager(f"{output_dir}/checkpoints")
    
    try:
        for epoch in range(start_epoch, config['training']['num_epochs']):
            # 收集数据
            trajectories = trainer.collect_trajectories(config['training']['steps_per_epoch'])
            
            # 更新策略
            metrics = trainer.update(trajectories)
            
            # 记录指标
            for name, value in metrics.items():
                writer.add_scalar(f'training/{name}', value, epoch)
            
            # 定期评估
            if (epoch + 1) % config['training']['eval_interval'] == 0:
                # 对抗随机玩家
                random_stats = evaluator.evaluate_against_random()
                for name, value in random_stats.items():
                    writer.add_scalar(f'evaluation/vs_random/{name}', value, epoch)
                
                # 自我对弈
                self_play_stats = evaluator.evaluate_self_play()
                for name, value in self_play_stats.items():
                    writer.add_scalar(f'evaluation/self_play/{name}', value, epoch)
                
                # 更新监控
                should_stop = monitor.update(epoch, {
                    'win_rate': random_stats.get('win_rate', 0),
                    **metrics
                })
                
                # 保存检查点
                is_best = epoch == monitor.best_epoch
                checkpoint_manager.save_checkpoint(
                    epoch,
                    model,
                    trainer.optimizer,
                    {
                        'training_metrics': metrics,
                        'evaluation_metrics': random_stats
                    },
                    is_best
                )
                
                if should_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # 可视化
            visualizer.render_game_state(
                env,
                save_path=f'{output_dir}/visualizations/game_state_epoch_{epoch+1}.png'
            )
            
            # 打印进度
            print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
            print(f"Training metrics: {metrics}")
            print(f"Vs Random: {random_stats}")
            print(f"Self Play: {self_play_stats}")
            print("-" * 50)
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise  # 添加这行以显示完整的错误堆栈
    finally:
        # 保存最终状态
        if 'epoch' in locals() and 'metrics' in locals() and 'random_stats' in locals():
            checkpoint_manager.save_checkpoint(
                epoch,
                model,
                trainer.optimizer,
                {
                    'training_metrics': metrics,
                    'evaluation_metrics': random_stats
                }
            )
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from', type=str, help='恢复训练的检查点路径')
    args = parser.parse_args()
    main()