import os
import torch
from typing import Dict, Any, Optional
import logging
from pathlib import Path
from torch import nn
from torch.optim import Optimizer

class CheckpointManager:
    """Manager for saving and loading model checkpoints"""
    
    def __init__(self, checkpoint_dir: str):
        """Initialize the checkpoint manager
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing checkpoint manager...")
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        metrics: Dict[str, float]
    ) -> None:
        """Save a checkpoint
        
        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer to save
            metrics: Training metrics to save
        """
        self.logger.info(f"Saving checkpoint for epoch {epoch+1}...")
        
        try:
            # Create checkpoint data
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(checkpoint, checkpoint_path)
            
            self.logger.info(f"Checkpoint saved successfully to {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            self.logger.exception("Checkpoint saving error details:")
            raise
            
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data if successful, None otherwise
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
        
        try:
            checkpoint = torch.load(checkpoint_path)
            self.logger.info("Checkpoint loaded successfully")
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            self.logger.exception("Checkpoint loading error details:")
            return None
            
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint
        
        Returns:
            Path to latest checkpoint if exists, None otherwise
        """
        self.logger.info("Looking for latest checkpoint...")
        
        try:
            checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if not checkpoints:
                self.logger.info("No checkpoints found")
                return None
                
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            self.logger.info(f"Latest checkpoint found: {latest_checkpoint}")
            return str(latest_checkpoint)
            
        except Exception as e:
            self.logger.error(f"Failed to find latest checkpoint: {str(e)}")
            self.logger.exception("Error details:")
            return None