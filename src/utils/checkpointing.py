import os
import torch
from typing import Dict, Any

class CheckpointManager:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def save_checkpoint(self, epoch: int, model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer, metrics: Dict[str, Any], 
                       is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics
        }
        
        # 保存最新的检查点
        filename = f"checkpoint_epoch_{epoch}.pt"
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
        
        # 如果是最佳模型，保存一个副本
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
        
    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        """加载检查点"""
        path = os.path.join(self.save_dir, filename)
        if os.path.exists(path):
            return torch.load(path)
        return None
        
    def list_checkpoints(self):
        """列出所有检查点"""
        return [f for f in os.listdir(self.save_dir) if f.endswith(".pt")]