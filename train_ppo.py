import multiprocessing
import psutil
import sys
import gc

# 设置多进程启动方法为 spawn
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # 如果已经设置过则忽略

import torch
import os
from datetime import datetime
import yaml
import argparse
import logging
from pathlib import Path

from src.models.network import RobotTurtlesNet
from src.environment.game_env import RobotTurtlesEnv, GameConfig
from src.training.ppo import PPOTrainer
from src.evaluation.evaluator import Evaluator
from src.visualization.visualizer import GameVisualizer
from torch.utils.tensorboard import SummaryWriter
from src.utils.checkpointing import CheckpointManager
from src.utils.training_monitor import TrainingMonitor

def get_memory_usage():
    """获取当前进程的内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转换为MB

def setup_logging(output_dir: str) -> None:
    """设置日志"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    root_logger.handlers = []
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(
        os.path.join(output_dir, 'train.log'),
        mode='w',
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    
    # 测试日志输出
    logging.info("日志系统初始化完成")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练机器海龟AI')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                      help='训练配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='训练设备')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'outputs/{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    setup_logging(str(output_dir))
    logger = logging.getLogger(__name__)
    logger.info("="*50)
    logger.info("开始训练...")
    logger.info(f"Python 版本: {sys.version}")
    logger.info(f"PyTorch 版本: {torch.__version__}")
    logger.info(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    logger.info(f"CPU 核心数: {multiprocessing.cpu_count()}")
    logger.info(f"初始内存使用: {get_memory_usage():.2f} MB")
    logger.info(f"配置: {config}")
    logger.info("="*50)
    
    # 保存配置
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # 初始化组件
    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    env_config = GameConfig(**config['environment'])
    env = RobotTurtlesEnv(env_config)
    logger.info(f"环境配置: {env_config}")
    
    model = RobotTurtlesNet(**config['model']).to(device)
    logger.info(f"模型结构:\n{model}")
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    trainer = PPOTrainer(
        model=model, 
        env=env, 
        device=device,
        learning_rate=float(config['ppo']['learning_rate']),
        clip_ratio=float(config['ppo']['epsilon']),
        value_coef=float(config['ppo']['value_coef']),
        entropy_coef=float(config['ppo']['entropy_coef']),
        epochs=int(config['ppo'].get('epochs', 10)),
        num_processes=int(config['training'].get('num_processes', 8)),
        parallel_envs=int(config['training'].get('parallel_envs', 16)),
        batch_size=int(config['training'].get('batch_size', 64)),
        num_workers=int(config['training'].get('num_workers', 4)),
        prefetch_factor=int(config['training'].get('prefetch_factor', 2)),
        pin_memory=bool(config['training'].get('pin_memory', True))
    )
    logger.info("训练器初始化完成")
    logger.info(f"当前内存使用: {get_memory_usage():.2f} MB")
    
    evaluator = Evaluator(model, device)
    visualizer = GameVisualizer()
    writer = SummaryWriter(output_dir / 'logs')
    checkpoint_manager = CheckpointManager(output_dir / 'checkpoints')
    training_monitor = TrainingMonitor(
        log_dir=str(output_dir / 'logs'),
        experiment_name="ppo_training"
    )

    # 恢复检查点
    start_epoch = 0
    if args.resume:
        checkpoint = checkpoint_manager.load_checkpoint(args.resume)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"从 epoch {start_epoch} 恢复训练")

    try:
        for epoch in range(start_epoch, config['training']['num_epochs']):
            logger.info("="*30 + f" Epoch {epoch+1}/{config['training']['num_epochs']} " + "="*30)
            logger.info(f"当前内存使用: {get_memory_usage():.2f} MB")
            
            # 轨迹收集
            logger.info("开始收集轨迹数据...")
            try:
                # 使用单进程收集轨迹，避免多进程带来的复杂性
                trajectories = trainer.collect_trajectories_single_process(config['training']['steps_per_epoch'])
                logger.info(f"收集了 {len(trajectories)} 条轨迹")
            except Exception as e:
                logger.error(f"收集轨迹失败: {str(e)}")
                logger.exception("轨迹收集错误详情:")
                break
            
            logger.info(f"收集后内存使用: {get_memory_usage():.2f} MB")
            
            # 更新策略
            logger.info("开始更新策略...")
            metrics = trainer.update(trajectories)
            
            # 记录指标
            logger.info(f"Epoch {epoch+1} 指标:")
            for name, value in metrics.items():
                writer.add_scalar(f'training/{name}', value, epoch)
                logger.info(f"  {name}: {value:.4f}")
            
            # 清理内存
            del trajectories
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"清理后内存使用: {get_memory_usage():.2f} MB")
            
            # 定期评估
            if (epoch + 1) % config['training']['eval_interval'] == 0:
                logger.info("-"*20 + " 开始评估 " + "-"*20)
                # 评估
                eval_stats = evaluator.evaluate(env)
                logger.info("评估结果:")
                for name, value in eval_stats.items():
                    writer.add_scalar(f'evaluation/{name}', value, epoch)
                    logger.info(f"  {name}: {value:.4f}")
                
                # 可视化
                visualizer.render_game_state(
                    env,
                    save_path=output_dir / f'visualizations/game_state_epoch_{epoch+1}.png'
                )
                
                # 检查是否需要早停
                should_stop = training_monitor.update(epoch, {
                    **metrics,
                    **eval_stats
                })
                
                # 保存检查点
                checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=trainer.optimizer,
                    metrics={**metrics, **eval_stats}
                )
                logger.info(f"保存了检查点: epoch_{epoch+1}")
                
                if should_stop:
                    logger.info(f"触发早停，在 epoch {epoch} 停止训练")
                    break

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.exception("训练失败")
        raise
    finally:
        writer.close()
        logger.info("训练结束")
        logger.info(f"最终内存使用: {get_memory_usage():.2f} MB")

if __name__ == '__main__':
    main()