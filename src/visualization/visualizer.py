import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns
from ..environment.game_env import RobotTurtlesEnv
import logging
import cv2

class GameVisualizer:
    """Visualizer for Robot Turtles game"""
    
    def __init__(self):
        """Initialize the visualizer"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing game visualizer...")
        
        # Initialize visualization settings
        self.board_size = 8
        self.cell_size = 60
        self.margin = 20
        
        # Calculate window size
        self.window_size = (
            self.board_size * self.cell_size + 2 * self.margin,
            self.board_size * self.cell_size + 2 * self.margin
        )
        
        self.logger.info("Visualizer initialized successfully")
        
    def render(self, board: np.ndarray, info: Dict[str, Any] = None) -> None:
        """Render the current game state
        
        Args:
            board: Game board state
            info: Additional game information
        """
        self.logger.info("Rendering game state...")
        
        # Create window
        window = np.ones((*self.window_size, 3), dtype=np.uint8) * 255
        
        # Draw grid
        for i in range(self.board_size + 1):
            # Vertical lines
            cv2.line(
                window,
                (self.margin + i * self.cell_size, self.margin),
                (self.margin + i * self.cell_size, self.window_size[1] - self.margin),
                (0, 0, 0),
                1
            )
            # Horizontal lines
            cv2.line(
                window,
                (self.margin, self.margin + i * self.cell_size),
                (self.window_size[0] - self.margin, self.margin + i * self.cell_size),
                (0, 0, 0),
                1
            )
        
        # Draw board elements
        for i in range(self.board_size):
            for j in range(self.board_size):
                x = self.margin + j * self.cell_size
                y = self.margin + i * self.cell_size
                
                # Draw player
                if board[0, i, j] == 1:
                    cv2.circle(
                        window,
                        (x + self.cell_size // 2, y + self.cell_size // 2),
                        self.cell_size // 3,
                        (0, 0, 255),
                        -1
                    )
                
                # Draw walls
                if board[1, i, j] == 1:
                    cv2.rectangle(
                        window,
                        (x + 5, y + 5),
                        (x + self.cell_size - 5, y + self.cell_size - 5),
                        (0, 255, 0),
                        -1
                    )
        
        # Draw info text if provided
        if info:
            text = f"Steps: {info.get('steps_taken', 0)}"
            cv2.putText(
                window,
                text,
                (self.margin, self.window_size[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        # Show window
        cv2.imshow('Robot Turtles', window)
        cv2.waitKey(1)
        
        self.logger.info("Game state rendered successfully")
        
    def close(self) -> None:
        """Close visualization window"""
        self.logger.info("Closing visualization window...")
        cv2.destroyAllWindows()
        self.logger.info("Visualization window closed")
        
    def render_game_state(
        self,
        env: RobotTurtlesEnv,
        save_path: str = None
    ):
        """渲染游戏状态"""
        plt.figure(figsize=(10, 10))
        
        # 绘制棋盘
        board = np.zeros((8, 8, 3))  # RGB
        
        # 填充棋盘内容
        for x in range(8):
            for y in range(8):
                content = env._get_cell_content((x, y))
                if content['empty']:
                    color = 'white'
                elif content['jewel']:
                    color = 'gold'
                elif content['wall'] is not None:
                    color = (
                        'gray' if content['wall'].value == 0 
                        else 'lightblue')
                elif content['player'] is not None:
                    player_idx = env.players.index(content['player'])
                    color = ['blue', 'red', 'green', 'yellow'][player_idx]
                    
                # 转换颜色到RGB
                rgb = plt.matplotlib.colors.to_rgb(color)
                board[y, x] = rgb
                
        plt.imshow(board)
        
        # 添加网格
        plt.grid(True)
        plt.xticks(range(8))
        plt.yticks(range(8))
        
        # 添加玩家信息
        info_text = ""
        for i, player in enumerate(env.players):
            info_text += f"Player {i}: pos={player.position}, dir={player.direction.name}\n"
            info_text += f"Score: {player.score}, Walls: S={player.stone_walls}, I={player.ice_walls}\n"
            if player.program:
                info_text += f"Program: {[card.name for card in player.program]}\n"
            info_text += "\n"
            
        plt.figtext(1.1, 0.5, info_text, fontsize=10, va='center')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        
    def plot_training_metrics(
        self,
        metrics: Dict[str, List[float]],
        save_path: str = None
    ):
        """绘制训练指标"""
        plt.figure(figsize=(15, 10))
        
        # 创建子图
        num_metrics = len(metrics)
        rows = (num_metrics + 1) // 2
        cols = min(num_metrics, 2)
        
        for i, (name, values) in enumerate(metrics.items()):
            plt.subplot(rows, cols, i+1)
            plt.plot(values)
            plt.title(name)
            plt.xlabel('Episode')
            plt.ylabel('Value')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_action_distribution(
        self,
        action_probs: np.ndarray,
        wall_probs: np.ndarray,
        save_path: str = None
    ):
        """绘制动作分布"""
        plt.figure(figsize=(15, 5))
        
        # 绘制动作概率
        plt.subplot(1, 2, 1)
        sns.barplot(x=range(len(action_probs)), y=action_probs)
        plt.title('Action Probabilities')
        plt.xlabel('Action Type')
        plt.ylabel('Probability')
        
        # 绘制墙放置概率热图
        plt.subplot(1, 2, 2)
        sns.heatmap(wall_probs, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Wall Placement Probabilities')
        
        if save_path:
            plt.savefig(save_path)
        plt.show() 