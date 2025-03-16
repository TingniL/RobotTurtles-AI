import gymnasium as gym
import numpy as np
from gymnasium import spaces
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict
import logging
from collections import deque

logger = logging.getLogger(__name__)

class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

class CardType(Enum):
    BLUE = 0    # 移动
    YELLOW = 1  # 转向
    PURPLE = 2  # 射线
    LASER = 3   # 激光
    BUG = 4     # BugCard

class WallType(Enum):
    STONE = 0  # 石墙（不可摧毁）
    ICE = 1    # 冰墙（可摧毁）

@dataclass
class GameConfig:
    """游戏配置"""
    num_players: int = 2
    enable_bug_card: bool = False  # 是否启用BugCard
    three_rounds: bool = False     # 是否启用三局制
    max_program_size: int = 20     # 最大程序长度

@dataclass
class Player:
    position: Tuple[int, int]
    direction: Direction
    stone_walls: int = 3  # 石墙数量
    ice_walls: int = 2    # 冰墙数量
    cards: List[CardType] = None
    start_position: Tuple[int, int] = None
    program: List[CardType] = None  # 程序
    hand: List[CardType] = None     # 手牌
    score: int = 0                  # 分数
    bug_card: bool = False          # BugCard
    program_reversed: bool = False   # 程序是否反转

    def __post_init__(self):
        if self.program is None:
            self.program = []
        if self.hand is None:
            self.hand = []
            if self.cards:
                self.draw_cards(5)

    def draw_cards(self, count: int):
        """从牌堆中抽取指定数量的牌"""
        drawn = []
        while len(drawn) < count and self.cards:
            drawn.append(self.cards.pop(0))
        self.hand.extend(drawn)
        return drawn

    def add_to_program(self, card: CardType):
        """添加卡牌到程序中"""
        if self.program_reversed:
            self.program.insert(0, card)
        else:
            self.program.append(card)

class RobotTurtlesEnv(gym.Env):
    """机器海龟游戏环境"""
    
    def __init__(self, config: GameConfig = None):
        super().__init__()
        self.config = config or GameConfig()
        self.num_players = self.config.num_players
        self.round = 1  # 当前回合数
        
        # 动作空间大小
        n_actions = 7  # 基础动作数量
        if self.config.enable_bug_card:
            n_actions += 1
        self.action_space = spaces.Discrete(n_actions)
        
        # 观察空间
        board_channels = 8  # 基础通道数
        if self.config.enable_bug_card:
            board_channels += 1
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(board_channels, 8, 8), dtype=np.float32),
            "hand": spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.float32),
            "program": spaces.Box(low=0, high=1, shape=(self.config.max_program_size, 5), dtype=np.float32)
        })
        
        # 初始化游戏状态
        self.board = None
        self.players = None
        self.current_player = 0
        self.game_over = False
        self.winner = None
        
        # 重置环境
        self.reset()

    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)
        
        # 初始化棋盘
        self.board = np.zeros((8, 8), dtype=int)
        
        # 初始化玩家
        self.players = []
        start_positions = [(0, 3), (7, 3), (3, 0), (3, 7)][:self.num_players]
        start_directions = [Direction.SOUTH, Direction.NORTH, Direction.EAST, Direction.WEST][:self.num_players]
        
        for i in range(self.num_players):
            # 创建卡牌堆
            cards = self._create_deck()
            
            # 创建玩家
            player = Player(
                position=start_positions[i],
                direction=start_directions[i],
                cards=cards,
                start_position=start_positions[i]
            )
            self.players.append(player)
            
            # 在棋盘上标记玩家位置
            self.board[start_positions[i]] = i + 1
        
        self.current_player = 0
        self.game_over = False
        self.winner = None
        
        return self._get_observation(), {}

    def _create_deck(self) -> List[CardType]:
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
        self.np_random.shuffle(deck)
        return deck

    def step(self, action):
        """执行一步动作"""
        if self.game_over:
            return self._get_observation(), 0, True, False, {}
        
        player = self.players[self.current_player]
        reward = 0
        
        # 处理动作
        if action < 5:  # 使用手牌
            if action < len(player.hand):
                card = player.hand.pop(action)
                player.add_to_program(card)
                reward = 0.1  # 小奖励
        elif action == 5:  # 执行程序
            reward = self._execute_program(player)
        elif action == 6:  # 清空程序
            player.program = []
            reward = -0.1  # 小惩罚
        elif action == 7 and self.config.enable_bug_card:  # 使用BugCard
            if player.bug_card:
                player.program_reversed = not player.program_reversed
                player.bug_card = False
                reward = 0.2
        
        # 检查游戏是否结束
        done = self.game_over
        
        # 更新当前玩家
        if not done:
            self.current_player = (self.current_player + 1) % self.num_players
        
        return self._get_observation(), reward, done, False, {}

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

    def _check_win_condition(self, player: Player) -> bool:
        """检查是否达到胜利条件"""
        # 简单实现：到达起始位置即胜利
        return player.position == player.start_position

    def _fire_laser(self, player: Player) -> Tuple[int, int]:
        """发射激光"""
        current_pos = player.position
        direction = player.direction
        
        while True:
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

    def _get_observation(self):
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
        if self.config.enable_bug_card:
            bug_channel = np.zeros((8, 8), dtype=np.float32)
            for i, player in enumerate(self.players):
                if player.bug_card:
                    x, y = player.position
                    bug_channel[x, y] = 1
            channels.append(bug_channel)
        
        # 手牌和程序的one-hot编码
        current_player = self.players[self.current_player]
        hand = np.zeros((5, 5), dtype=np.float32)
        program = np.zeros((self.config.max_program_size, 5), dtype=np.float32)
        
        for i, card in enumerate(current_player.hand):
            if i < 5:
                hand[i, card.value] = 1
                
        for i, card in enumerate(current_player.program):
            if i < self.config.max_program_size:
                program[i, card.value] = 1
        
        return {
            "board": np.stack(channels),
            "hand": hand,
            "program": program
        }

    def render(self, mode="human"):
        """渲染游戏状态"""
        if mode == "human":
            # 简单的控制台渲染
            for i in range(8):
                for j in range(8):
                    if self.board[i, j] > 0:
                        print(f"P{self.board[i, j]}", end=" ")
                    elif self.board[i, j] == -1:
                        print("S", end=" ")  # Stone wall
                    elif self.board[i, j] == -2:
                        print("I", end=" ")  # Ice wall
                    else:
                        print(".", end=" ")
                print()
            print()
