import random
from typing import List, Optional, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...graph import IMGraph

random.seed(123)
np.random.seed(123)


class IMEnvironment:
    """影响力最大化强化学习环境。

    用于训练深度学习影响力最大化算法，提供状态管理和奖励计算。

    Attributes:
        graph: 输入图对象。
        budget: 种子节点预算。
        method: 奖励计算方法，支持 'MC' (蒙特卡洛) 或 'RR' (反向可达)。
        state: 当前状态向量，1 表示已选中，0 表示未选中。
    """

    def __init__(
        self,
        graph: 'IMGraph',
        budget: int,
        method: str = 'RR',
        num_trials: int = 10000
    ):
        """初始化环境。

        Args:
            graph: 输入图对象。
            budget: 种子节点预算。
            method: 奖励计算方法，支持 'MC' 或 'RR'，默认为 'RR'。
            num_trials: 蒙特卡洛模拟次数或 RR 集合数量，默认为 10000。
        """
        self.graph = graph
        self.budget = budget
        self.method = method
        self.num_trials = num_trials

        self.state: List[int] = []
        self.prev_inf: float = 0.0
        self.states: List[List[int]] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []

        self._rr_sets: List[List[int]] = []

    def reset(self):
        """重置环境状态。"""
        self.state = [0] * self.graph.num_nodes
        self.prev_inf = 0.0
        self.states = []
        self.actions = []
        self.rewards = []

        if self.method == 'RR':
            self._rr_sets = self._generate_rr_sets()

    def _generate_rr_sets(self) -> List[List[int]]:
        """生成反向可达集合（使用 C++ 实现）。

        Returns:
            List[List[int]]: RR 集合列表。
        """
        from pynetim.utils import generate_rr_sets
        return generate_rr_sets(self.graph, self.num_trials, model='IC')

    def _compute_influence_mc(self, seeds: List[int]) -> float:
        """使用蒙特卡洛方法计算影响力（使用 C++ 传播模型）。

        Args:
            seeds: 种子节点列表。

        Returns:
            float: 估计的影响力值。
        """
        from pynetim.diffusion_model import IndependentCascadeModel
        
        model = IndependentCascadeModel(self.graph, set(seeds))
        return model.run_monte_carlo_diffusion(self.num_trials, use_multithread=True, num_threads=4)

    def _compute_influence_rr(self, seeds: List[int]) -> float:
        """使用反向可达方法计算影响力。

        Args:
            seeds: 种子节点列表。

        Returns:
            float: 估计的影响力值。
        """
        seeds_set = set(seeds)
        covered = sum(1 for rr_set in self._rr_sets if seeds_set & set(rr_set))
        return covered / len(self._rr_sets) * self.graph.num_nodes

    def compute_reward(self, seeds: List[int]) -> float:
        """计算奖励值。

        Args:
            seeds: 当前种子节点列表。

        Returns:
            float: 奖励值（边际影响力增益）。
        """
        if self.method == 'MC':
            influence = self._compute_influence_mc(seeds)
        elif self.method == 'RR':
            influence = self._compute_influence_rr(seeds)
        else:
            raise ValueError(f"不支持的方法: {self.method}")

        reward = influence - self.prev_inf
        self.prev_inf = influence
        return reward

    def step(self, node: int) -> tuple:
        """执行一步动作。

        Args:
            node: 选择的节点。

        Returns:
            tuple: (reward, done) 奖励值和是否结束。
        """
        if self.state[node] == 1:
            return 0.0, False

        self.states.append(self.state.copy())
        self.actions.append(node)
        self.state[node] = 1

        done = len(self.actions) >= self.budget
        reward = self.compute_reward(self.actions) if done else 0.0
        self.rewards.append(reward)

        return reward, done

    def get_state(self) -> List[int]:
        """获取当前状态。

        Returns:
            List[int]: 状态向量。
        """
        return self.state.copy()

    def get_selected_nodes(self) -> List[int]:
        """获取已选择的节点列表。

        Returns:
            List[int]: 已选择的节点列表。
        """
        return self.actions.copy()
