import time
from typing import Dict, Any, List
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import defaultdict

class TrainingMonitor:
    """Monitor for training progress and metrics"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """Initialize the training monitor
        
        Args:
            log_dir: Directory to store logs
            experiment_name: Name of the training experiment
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing training monitor...")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.metrics = defaultdict(list)
        
        self.logger.info(f"Training monitor initialized for experiment: {experiment_name}")
        
    def update(self, metrics: Dict[str, float]) -> None:
        """Update training metrics
        
        Args:
            metrics: Dictionary of metric names and values
        """
        self.logger.info("Updating training metrics...")
        
        try:
            for name, value in metrics.items():
                self.metrics[name].append(value)
                
            self.logger.info("Metrics updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {str(e)}")
            self.logger.exception("Error details:")
            
    def save_metrics(self) -> None:
        """Save current metrics to file"""
        self.logger.info("Saving training metrics...")
        
        try:
            metrics_file = self.log_dir / f'{self.experiment_name}_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(dict(self.metrics), f, indent=4)
                
            self.logger.info(f"Metrics saved successfully to {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {str(e)}")
            self.logger.exception("Error details:")
            
    def plot_metrics(self) -> None:
        """Plot training metrics"""
        self.logger.info("Plotting training metrics...")
        
        try:
            # Create figure with subplots
            num_metrics = len(self.metrics)
            fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4*num_metrics))
            if num_metrics == 1:
                axes = [axes]
                
            # Plot each metric
            for ax, (name, values) in zip(axes, self.metrics.items()):
                ax.plot(values, label=name)
                ax.set_title(f'Training {name}')
                ax.set_xlabel('Step')
                ax.set_ylabel('Value')
                ax.grid(True)
                ax.legend()
                
            # Save plot
            plt.tight_layout()
            plot_file = self.log_dir / f'{self.experiment_name}_metrics.png'
            plt.savefig(plot_file)
            plt.close()
            
            self.logger.info(f"Metrics plot saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to plot metrics: {str(e)}")
            self.logger.exception("Error details:")
            
    def get_metrics(self) -> Dict[str, List[float]]:
        """Get current metrics
        
        Returns:
            Dictionary of metric names and their values
        """
        return dict(self.metrics)
        
    def clear(self) -> None:
        """Clear all metrics"""
        self.logger.info("Clearing training metrics...")
        self.metrics.clear()
        self.logger.info("Metrics cleared successfully")
