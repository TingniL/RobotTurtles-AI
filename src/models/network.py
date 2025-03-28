import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

class RobotTurtlesNet(nn.Module):
    """Neural network for Robot Turtles game"""
    
    def __init__(self, input_channels: int = 9):
        """Initialize the network
        
        Args:
            input_channels: Number of input channels
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Robot Turtles network...")
        
        super().__init__()
        
        # CNN for board processing
        self.board_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size
        self.board_flat_size = 64 * 8 * 8
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.board_flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Action type output
        self.action_type = nn.Linear(128, 8)
        
        # Wall placement output
        self.wall_probs = nn.Linear(128, 64)  # 8x8 grid
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.board_flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.logger.info("Network initialized successfully")
        
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass
        
        Args:
            x: Input dictionary containing board state
            
        Returns:
            Dictionary containing action probabilities and value
        """
        # Process board state
        board = x['board']
        board_features = self.board_conv(board)
        
        # Get action probabilities
        policy_features = self.policy_head(board_features)
        action_probs = F.softmax(self.action_type(policy_features), dim=-1)
        wall_probs = F.softmax(self.wall_probs(policy_features), dim=-1)
        
        # Get state value
        value = self.value_head(board_features)
        
        return {
            'action_probs': action_probs,
            'wall_probs': wall_probs,
            'value': value
        }