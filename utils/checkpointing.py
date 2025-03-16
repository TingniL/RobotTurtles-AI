import torch
import os
import json
from typing import Dict, Any

class CheckpointManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, Any],
        is_best: bool = False
    ):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.checkpoints_dir, 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        # 保存定期检查点
        if (epoch + 1) % 100 == 0:
            epoch_path = os.path.join(self.checkpoints_dir, f'epoch_{epoch+1}.pt')
            torch.save(checkpoint, epoch_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.checkpoints_dir, 'best.pt')
            torch.save(checkpoint, best_path)
            
        # 保存训练进度
        progress = {
            'epoch': epoch,
            'metrics': metrics
        }
        progress_path = os.path.join(self.output_dir, 'progress.json')
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2) 