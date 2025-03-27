import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, TypedDict
import logging
from collections import deque

# 配置日志
logger = logging.getLogger(__name__)

class Action(TypedDict):
    """动作类型"""
    action_type: int
    wall_position: Tuple[int, int]

class Observation(TypedDict):
    """观察类型"""
    board: np.ndarray
    hand: np.ndarray
    program: np.ndarray

class CardType(Enum):
    """卡牌类型"""
    BLUE = 0    # 移动卡
    YELLOW = 1  # 转向卡
    PURPLE = 2  # 射线卡
    LASER = 3   # 激光卡
    BUG = 4     # Bug卡

class Direction(Enum):
    """方向枚举"""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

class WallType(Enum):
    """墙的类型"""
    STONE = -1  # 石墙
    ICE = -2    # 冰墙

@dataclass
class GameConfig:
    """游戏配置"""
    num_players: int = 2
    enable_bug_card: bool = True
    max_program_size: int = 20
    board_size: int = 8
    three_rounds: bool = False  # 是否启用三回合制

class Player:
    """玩家类"""
    def __init__(self, id: int = 0, position: Tuple[int, int] = None, direction: Direction = None):
        self.id = id
        self.start_position = position if position is not None else (0, 0)
        self.position = position if position is not None else (0, 0)
        self.direction = direction if direction is not None else Direction.NORTH
        self.hand = []  # 手牌
        self.program = []  # 程序
        self.stone_walls = 3  # 石墙数量
        self.ice_walls = 2   # 冰墙数量
        self.bug_card = True  # 是否有Bug卡
        self.program_reversed = False  # 程序是否反转
        
    def add_to_program(self, card: CardType):
        """添加卡牌到程序"""
        if len(self.program) < 20:  # 限制程序长度
            self.program.append(card)

class RobotTurtlesEnv(gym.Env):
    """机器海龟游戏环境"""
    
    def __init__(self, config: Optional[GameConfig] = None):
        """
        初始化游戏环境
        
        Args:
            config: 游戏配置
        """
        super().__init__()
        
        # 设置配置
        if config is None:
            config = GameConfig()
        self.config = config
        
        # 游戏状态
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.current_player = 0
        self.current_turn = 0  # 添加回合计数器
        self.num_players = config.num_players
        self.players = []
        self.game_over = False
        self.winner = None
        
        # 设置动作空间和观察空间
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(8),  # 使用卡牌0-3, 放置墙4-5, 执行程序6, 使用BugCard7
            'wall_position': spaces.Tuple((spaces.Discrete(8), spaces.Discrete(8)))  # 墙的放置位置
        })
        
        # 观察空间
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=1, shape=(9, 8, 8)),
            'hand': spaces.Box(low=0, high=1, shape=(5, 5)),
            'program': spaces.Box(low=0, high=1, shape=(config.max_program_size, 5))
        })
        
        # 重置环境
        self.reset()

    def reset(self, seed=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """重置游戏环境
        
        Args:
            seed: 随机种子
            
        Returns:
            观察和信息字典
        """
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            
        # 重置游戏状态
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.current_player = 0
        self.current_turn = 0  # 重置回合计数
        self.game_over = False
        self.winner = None
        
        # 创建牌组
        self.deck = self._create_deck()
        
        # 创建玩家
        self.players = []
        for i in range(self.num_players):
            # 创建玩家
            player = Player(
                id=i,
                position=None,  # 稍后设置
                direction=Direction(np.random.randint(0, 4))
            )
            
            # 放置玩家位置
            while True:
                x, y = np.random.randint(0, 8, size=2)
                if self.board[x, y] == 0:  # 位置为空
                    player.position = (x, y)
                    player.start_position = (x, y)  # 设置起始位置
                    self.board[x, y] = i + 1  # 玩家编号从1开始
                    break
            
            # 发牌
            for _ in range(5):
                if self.deck:
                    player.hand.append(self.deck.popleft())
            
            # 设置墙
            player.stone_walls = 3
            player.ice_walls = 2
            
            # 设置BugCard
            player.bug_card = self.config.enable_bug_card
            
            self.players.append(player)
            
        return self._get_observation(), {}

    def _create_deck(self) -> deque:
        """创建一副牌"""
        deck = []
        # 添加基础卡牌
        deck.extend([CardType.BLUE] * 18)    # 移动卡
        deck.extend([CardType.YELLOW] * 8)   # 转向卡
        deck.extend([CardType.PURPLE] * 3)   # 射线卡
        deck.extend([CardType.LASER] * 3)    # 激光卡
        
        # 如果启用了BugCard
        if self.config.enable_bug_card:
            deck.extend([CardType.BUG] * 3)
        
        # 洗牌
        np.random.shuffle(deck)
        return deque(deck)

    def step(self, action: dict) -> Tuple[dict, float, bool, bool, dict]:
        """执行一步动作
        
        Args:
            action: dict - 包含动作类型和墙放置位置的字典
            
        Returns:
            observation: 当前观察
            reward: 奖励值
            terminated: 游戏是否结束
            truncated: 回合是否被截断
            info: 额外信息
        """
        if self.game_over:
            return self._get_observation(), 0, True, False, {}
        
        player = self.players[self.current_player]
        reward = 0
        
        # 处理动作
        action_type = action['action_type']
        wall_position = action.get('wall_position', (0, 0))
        
        if action_type < 4:  # 使用卡牌
            if action_type < len(player.hand):
                card = player.hand.pop(action_type)
                player.add_to_program(card)
                reward = 0.1  # 小奖励
        elif action_type == 4:  # 放置石墙
            if player.stone_walls > 0 and self._is_valid_wall_position(wall_position):
                self.board[wall_position] = WallType.STONE.value
                player.stone_walls -= 1
                reward = 0.1
        elif action_type == 5:  # 放置冰墙
            if player.ice_walls > 0 and self._is_valid_wall_position(wall_position):
                self.board[wall_position] = WallType.ICE.value
                player.ice_walls -= 1
                reward = 0.1
        elif action_type == 6:  # 执行程序
            if player.program:
                program_reward = self._execute_program(player)
                reward += program_reward
        elif action_type == 7:  # 使用Bug卡
            if player.bug_card:
                player.program_reversed = not player.program_reversed
                player.bug_card = False
                reward = 0.1
        
        # 检查游戏是否结束
        if self._check_win_condition(player):
            self.game_over = True
            self.winner = self.current_player
            reward = 1.0
        
        # 切换到下一个玩家
        self.current_player = (self.current_player + 1) % self.num_players
        
        # 增加回合计数
        if self.current_player == 0:  # 一轮结束
            self.current_turn += 1
            
            # 检查是否超过最大回合数（防止游戏无限进行）
            if self.current_turn >= 100:  # 设置一个合理的最大回合数
                self.game_over = True
                reward = 0  # 平局情况
        
        return self._get_observation(), reward, self.game_over, False, {
            'winner': self.winner,
            'current_player': self.current_player,
            'current_turn': self.current_turn
        }

    def _execute_program(self, player: Player) -> float:
        """执行玩家的程序"""
        if not player.program:
            return -0.1
        
        reward = 0
        original_position = player.position
        
        for card in player.program:
            if card == CardType.BLUE:  # 移动
                new_pos = self._get_next_position(player.position, player.direction)
                if self._is_valid_move(new_pos):
                    self.board[player.position] = 0
                    player.position = new_pos
                    self.board[new_pos] = self.current_player + 1
                    reward += 0.1
                # 移动失败不做特殊处理
            elif card == CardType.YELLOW:  # 转向
                player.direction = Direction((player.direction.value + 1) % 4)
                reward += 0.05
            elif card == CardType.PURPLE:  # 射线
                if self._check_win_condition(player):
                    self.game_over = True
                    self.winner = self.current_player
                    return 1.0
            elif card == CardType.LASER:  # 激光
                hit_pos = self._fire_laser(player)
                if hit_pos:
                    reward += 0.2
        
        # 清空程序
        player.program = []
        
        # 如果位置没有变化，给予负奖励
        if player.position == original_position:
            reward -= 0.1
        
        return reward

    def _get_next_position(self, position: Tuple[int, int], direction: Direction) -> Tuple[int, int]:
        """获取下一个位置"""
        x, y = position
        if direction == Direction.NORTH:
            return (x - 1, y)
        elif direction == Direction.SOUTH:
            return (x + 1, y)
        elif direction == Direction.EAST:
            return (x, y + 1)
        else:  # WEST
            return (x, y - 1)

    def _is_valid_move(self, position: Tuple[int, int]) -> bool:
        """检查移动是否有效"""
        x, y = position
        if x < 0 or x >= 8 or y < 0 or y >= 8:
            return False
        if self.board[position] != 0:  # 位置被占用
            return False
        return True

    def _is_valid_wall_position(self, position: Tuple[int, int]) -> bool:
        """检查墙的放置位置是否有效"""
        x, y = position
        # 检查是否在边界内
        if x < 0 or x >= 8 or y < 0 or y >= 8:
            return False
        # 检查位置是否已被占用
        if self.board[position] != 0:
            return False
        return True

    def _check_win_condition(self, player: Player) -> bool:
        """检查是否达到胜利条件"""
        # 简单实现：到达起始位置即胜利
        return player.position == player.start_position

    def _fire_laser(self, player: Player) -> Tuple[int, int]:
        """发射激光"""
        current_pos = player.position
        direction = player.direction
        max_steps = 8  # 最大步数限制，防止无限循环
        steps = 0
        
        while steps < max_steps:
            steps += 1
            next_pos = self._get_next_position(current_pos, direction)
            x, y = next_pos
            
            # 检查是否出界
            if x < 0 or x >= 8 or y < 0 or y >= 8:
                break
                
            # 检查是否击中墙
            if self.board[next_pos] < 0:  # 墙的标记为负数
                self.board[next_pos] = 0  # 摧毁墙
                return next_pos
                
            # 检查是否击中玩家
            if self.board[next_pos] > 0:
                return next_pos
                
            current_pos = next_pos
        
        return None

    def get_valid_actions(self) -> List[Dict[str, Any]]:
        """获取当前可用的动作列表"""
        valid_actions = []
        player = self.players[self.current_player]
        
        # 使用手牌的动作
        for i in range(len(player.hand)):
            valid_actions.append({
                'action_type': i,
                'wall_position': (0, 0)  # 使用手牌不需要墙的位置
            })
            
        # 放置石墙 - 只在玩家周围放置墙，而不是整个棋盘
        if player.stone_walls > 0:
            # 获取玩家周围的有效位置
            px, py = player.position
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                wall_pos = (px + dx, py + dy)
                if 0 <= wall_pos[0] < 8 and 0 <= wall_pos[1] < 8 and self._is_valid_wall_position(wall_pos):
                    valid_actions.append({
                        'action_type': 4,
                        'wall_position': wall_pos
                    })
                        
        # 放置冰墙 - 同样只在玩家周围放置
        if player.ice_walls > 0:
            px, py = player.position
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                wall_pos = (px + dx, py + dy)
                if 0 <= wall_pos[0] < 8 and 0 <= wall_pos[1] < 8 and self._is_valid_wall_position(wall_pos):
                    valid_actions.append({
                        'action_type': 5,
                        'wall_position': wall_pos
                    })
                        
        # 执行程序
        if len(player.program) > 0:
            valid_actions.append({
                'action_type': 6,
                'wall_position': (0, 0)
            })
            
        # 使用BugCard
        if self.config.enable_bug_card and player.bug_card:
            valid_actions.append({
                'action_type': 7,
                'wall_position': (0, 0)
            })
            
        # 如果没有有效动作，至少允许执行空程序
        if not valid_actions:
            valid_actions.append({
                'action_type': 6,
                'wall_position': (0, 0)
            })
            
        return valid_actions

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取观察空间"""
        # 创建多通道观察
        channels = []
        
        # 玩家位置通道
        for i in range(self.num_players):
            channel = np.zeros((8, 8), dtype=np.float32)
            channel[self.board == (i + 1)] = 1
            channels.append(channel)
        
        # 墙通道
        stone_wall_channel = np.zeros((8, 8), dtype=np.float32)
        ice_wall_channel = np.zeros((8, 8), dtype=np.float32)
        stone_wall_channel[self.board == -1] = 1
        ice_wall_channel[self.board == -2] = 1
        channels.extend([stone_wall_channel, ice_wall_channel])
        
        # 方向通道
        direction_channel = np.zeros((8, 8, 4), dtype=np.float32)
        for player in self.players:
            x, y = player.position
            direction_channel[x, y, player.direction.value] = 1
        channels.extend([direction_channel[:, :, i] for i in range(4)])
        
        # BugCard通道
        bug_channel = np.zeros((8, 8), dtype=np.float32)
        for i, player in enumerate(self.players):
            if player.bug_card:
                x, y = player.position
                bug_channel[x, y] = 1
        channels.append(bug_channel)
        
        # 手牌和程序的one-hot编码
        current_player = self.players[self.current_player]
        hand = np.zeros((5, 5), dtype=np.float32)  # 5张手牌，5种卡牌类型
        program = np.zeros((self.config.max_program_size, 5), dtype=np.float32)  # 程序长度，5种卡牌类型
        
        for i, card in enumerate(current_player.hand):
            if i < 5:
                hand[i, card.value] = 1
                
        for i, card in enumerate(current_player.program):
            if i < self.config.max_program_size:
                program[i, card.value] = 1
        
        stacked_channels = np.stack(channels)
        
        return {
            'board': stacked_channels,  # 9个通道：2个玩家位置 + 2个墙 + 4个方向 + 1个BugCard
            'hand': hand,
            'program': program
        }

    def render(self, mode="human"):
        """渲染游戏状态"""
        if mode == "human" and logger.level <= logging.INFO:
            # 只有在INFO级别或更低级别才进行渲染
            # 创建一个简单的字符串表示
            board_str = "\n"
            for i in range(8):
                row = ""
                for j in range(8):
                    if self.board[i, j] > 0:
                        row += f"P{int(self.board[i, j])} "
                    elif self.board[i, j] == -1:
                        row += "S  "  # Stone wall
                    elif self.board[i, j] == -2:
                        row += "I  "  # Ice wall
                    else:
                        row += ".  "
                board_str += row + "\n"
            
            # 只打印基本信息
            logger.debug(f"\n游戏状态: 当前玩家={self.current_player}, 回合={self.current_turn}")
            logger.debug(board_str)
            
            # 仅在DEBUG级别显示详细玩家信息
            if logger.level <= logging.DEBUG:
                for i, player in enumerate(self.players):
                    logger.debug(f"玩家{i}: 位置={player.position}, 方向={player.direction.name}")
                    logger.debug(f"手牌: {len(player.hand)}张, 程序: {len(player.program)}步")
        
        # 返回棋盘状态的字符表示（可用于其他渲染模式）
        return self.board.copy()
