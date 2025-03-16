import torch.multiprocessing as mp
import numpy as np
from typing import List, Dict
from ..environment.game_env import RobotTurtlesEnv, GameConfig

class ParallelDataGenerator:
    """并行数据生成器"""
    
    def __init__(self, num_processes: int = 4):
        self.num_processes = num_processes
        
    def generate_data(
        self,
        num_games: int,
        config: GameConfig = None
    ) -> List[Dict]:
        """并行生成游戏数据"""
        games_per_process = num_games // self.num_processes
        
        # 创建进程池
        with mp.Pool(self.num_processes) as pool:
            results = pool.starmap(
                self._generate_games,
                [(games_per_process, config) for _ in range(self.num_processes)]
            )
            
        # 合并结果
        all_data = []
        for result in results:
            all_data.extend(result)
            
        return all_data
    
    @staticmethod
    def _generate_games(
        num_games: int,
        config: GameConfig
    ) -> List[Dict]:
        """在单个进程中生成游戏数据"""
        env = RobotTurtlesEnv(config)
        game_data = []
        
        for _ in range(num_games):
            obs = env.reset()[0]
            done = False
            
            while not done:
                # 随机选择动作
                action = {
                    'action_type': env.action_space['action_type'].sample(),
                    'wall_position': (
                        np.random.randint(8),
                        np.random.randint(8)
                    )
                }
                
                next_obs, reward, done, _, _ = env.step(action)
                
                game_data.append({
                    'state': obs,
                    'action': action,
                    'reward': reward,
                    'next_state': next_obs,
                    'done': done
                })
                
                obs = next_obs
                
        return game_data 