import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns
from ..environment.game_env import RobotTurtlesEnv

class GameVisualizer:
    """游戏可视化工具"""
    
    def __init__(self):
        self.colors = {
            'empty': 'white',
            'player': ['blue', 'red', 'green', 'yellow'],
            'jewel': 'gold',
            'stone_wall': 'gray',
            'ice_wall': 'lightblue',
            'laser': 'red'
        }
        
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
                    color = self.colors['empty']
                elif content['jewel']:
                    color = self.colors['jewel']
                elif content['wall'] is not None:
                    color = (self.colors['stone_wall'] 
                            if content['wall'].value == 0 
                            else self.colors['ice_wall'])
                elif content['player'] is not None:
                    player_idx = env.players.index(content['player'])
                    color = self.colors['player'][player_idx]
                    
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