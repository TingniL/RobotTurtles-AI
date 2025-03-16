from typing import List, Dict
import numpy as np

class TrainingMonitor:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = -np.inf
        self.wait = 0
        self.best_epoch = 0
        self.history: List[Dict] = []
        
    def update(self, epoch: int, metrics: Dict) -> bool:
        """
        更新监控状态，返回是否应该提前停止
        """
        self.history.append({
            'epoch': epoch,
            **metrics
        })
        
        # 使用胜率作为主要指标
        current_value = metrics.get('win_rate', 0)
        
        if current_value > self.best_value + self.min_delta:
            self.best_value = current_value
            self.wait = 0
            self.best_epoch = epoch
            return False
        
        self.wait += 1
        return self.wait >= self.patience
        
    def get_best_metrics(self) -> Dict:
        """获取最佳epoch的指标"""
        return self.history[self.best_epoch] 