import multiprocessing
import psutil
import sys
import gc
import time

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
    """Setup logging configuration"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(
        os.path.join(output_dir, 'train.log'),
        mode='w',
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set third-party library log levels
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    
    # Test log output
    logging.info("Logging system initialized")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Robot Turtles AI')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                      help='Path to training configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint for resuming training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Training device')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'outputs/{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(str(output_dir))
    logger = logging.getLogger(__name__)
    logger.info("="*50)
    logger.info("Starting training...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CPU cores: {multiprocessing.cpu_count()}")
    logger.info(f"Initial memory usage: {get_memory_usage():.2f} MB")
    logger.info(f"Configuration: {config}")
    logger.info("="*50)
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Initialize components
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    env_config = GameConfig(**config['environment'])
    env = RobotTurtlesEnv(env_config)
    logger.info(f"Environment configuration: {env_config}")
    
    model = RobotTurtlesNet(**config['model']).to(device)
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")
    
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
    logger.info("Trainer initialized")
    logger.info(f"Current memory usage: {get_memory_usage():.2f} MB")
    
    evaluator = Evaluator(model, device)
    visualizer = GameVisualizer()
    writer = SummaryWriter(output_dir / 'logs')
    checkpoint_manager = CheckpointManager(output_dir / 'checkpoints')
    training_monitor = TrainingMonitor(
        log_dir=str(output_dir / 'logs'),
        experiment_name="ppo_training"
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = checkpoint_manager.load_checkpoint(args.resume)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming training from epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, config['training']['num_epochs']):
            logger.info("="*30 + f" Epoch {epoch+1}/{config['training']['num_epochs']} " + "="*30)
            logger.info(f"Current memory usage: {get_memory_usage():.2f} MB")
            
            # Collect trajectories
            logger.info("Starting trajectory collection...")
            try:
                # Use single process collection to avoid complexity
                start_time = time.time()
                trajectories = trainer.collect_trajectories_single_process(config['training']['steps_per_epoch'])
                collect_time = time.time() - start_time
                logger.info(f"Collected {len(trajectories)} trajectories in {collect_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Failed to collect trajectories: {str(e)}")
                logger.exception("Trajectory collection error details:")
                break
            
            logger.info(f"Memory usage after collection: {get_memory_usage():.2f} MB")
            
            # Update policy
            logger.info("Starting policy update...")
            try:
                start_time = time.time()
                # Use simplified update method for better stability
                metrics = trainer.update_simplified(trajectories)
                update_time = time.time() - start_time
                logger.info(f"Update completed in {update_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Failed to update policy: {str(e)}")
                logger.exception("Policy update error details:")
                metrics = {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy_loss': 0.0,
                    'total_loss': 0.0
                }
            
            # Record metrics
            logger.info(f"Epoch {epoch+1} metrics:")
            for name, value in metrics.items():
                writer.add_scalar(f'training/{name}', value, epoch)
                logger.info(f"  {name}: {value:.4f}")
            
            # Clean up memory
            del trajectories
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Memory usage after cleanup: {get_memory_usage():.2f} MB")
            
            # Periodic evaluation
            if (epoch + 1) % config['training']['eval_interval'] == 0:
                logger.info("-"*20 + " Starting Evaluation " + "-"*20)
                try:
                    # Evaluate
                    eval_stats = evaluator.evaluate_against_random(
                        num_games=config['evaluation']['num_episodes']
                    )
                    logger.info("Evaluation results:")
                    for name, value in eval_stats.items():
                        writer.add_scalar(f'evaluation/{name}', value, epoch)
                        logger.info(f"  {name}: {value:.4f}")
                except Exception as e:
                    logger.error(f"Evaluation failed: {str(e)}")
                    logger.exception("Evaluation error details:")
                    eval_stats = {}
                
                # Save checkpoint
                try:
                    checkpoint_manager.save_checkpoint(
                        epoch=epoch,
                        model=model,
                        optimizer=trainer.optimizer,
                        metrics={**metrics, **eval_stats}
                    )
                    logger.info(f"Saved checkpoint: epoch_{epoch+1}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {str(e)}")
                    logger.exception("Checkpoint saving error details:")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.exception("Training failed")
        raise
    finally:
        writer.close()
        logger.info("Training completed")
        logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")

if __name__ == '__main__':
    main()