import random
from typing import List, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ....graph import IMGraph

random.seed(123)
np.random.seed(123)


class GraphEnvironment:
    """BiGDN 强化学习环境（内部使用，不向外导出）。

    References:
        BiGDN: An end-to-end influence maximization framework based on deep reinforcement
        learning and graph neural networks.
        Wenlong Zhu, Kaijing Zhang, Jiahui Zhong, Chengle Hou, Jie Ji.
        Expert Systems with Applications, 270:126384, 2025.
    """

    def __init__(
        self,
        graphs: List['IMGraph'],
        k: int,
        gamma: float = 0.99,
        n_steps: int = 2,
        method: str = 'MC',
        num_trials: int = 1000
    ):
        self.graphs = graphs
        self.k = k
        self.gamma = gamma
        self.n_steps = n_steps
        self.method = method
        self.num_trials = num_trials
        self.graph = None
        self.state = None
        self.preview_reward = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.seeds = []
        self.state_records = {}

    def reset(self) -> List[int]:
        """重置环境。"""
        self.graph = random.choice(self.graphs)
        self.seeds = []
        self.state = [0] * self.graph.num_nodes
        self.preview_reward = 0
        self.states = []
        self.actions = []
        self.rewards = []
        return self.state

    def step(self, action: int) -> tuple:
        """执行动作，转移到新状态。"""
        self.states.append(self.state.copy())
        self.state[action] = 1
        self.seeds.append(action)
        reward = self.compute_reward()

        done = False
        if len(self.seeds) == self.k:
            done = True

        if done:
            self.states.append(self.state.copy())

        self.actions.append(action)
        self.rewards.append(reward)
        return reward, self.state, done

    def compute_reward(self) -> float:
        """计算奖励值。"""
        str_seeds = str(id(self.graph)) + str(sorted(self.seeds))
        if self.method == 'MC':
            if str_seeds in self.state_records:
                current_reward = self.state_records[str_seeds]
            else:
                current_reward = self._compute_influence_mc(self.seeds)
            r = current_reward - self.preview_reward
            self.preview_reward = current_reward
            self.state_records[str_seeds] = current_reward
            return r
        elif self.method == 'RR':
            if str_seeds in self.state_records:
                current_reward = self.state_records[str_seeds]
            else:
                current_reward = self._compute_influence_rr(self.seeds)
            r = current_reward - self.preview_reward
            self.preview_reward = current_reward
            self.state_records[str_seeds] = current_reward
            return r
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _compute_influence_mc(self, seeds: List[int]) -> float:
        """使用蒙特卡洛方法计算影响力（使用 C++ 传播模型）。"""
        from pynetim.diffusion_model import IndependentCascadeModel

        model = IndependentCascadeModel(self.graph, set(seeds))
        return model.run_monte_carlo_diffusion(self.num_trials, use_multithread=True, num_threads=4)

    def _compute_influence_rr(self, seeds: List[int]) -> float:
        """使用 RR 集合计算影响力。"""
        seeds_set = set(seeds)
        covered = sum(1 for rr_set in self._rr_sets if seeds_set & set(rr_set))
        return covered / len(self._rr_sets) * self.graph.num_nodes

    def n_step_add_buffer(self, buffer):
        """将 n 步经验添加到回放缓冲区。"""
        states = self.states
        rewards = self.rewards
        n = self.n_steps
        gamma = self.gamma

        for i in range(len(states) - n):
            done = (i + n) == (len(states) - 1)
            next_state = states[i + n]

            n_reward = sum(rewards[i + j] * (gamma ** j) for j in range(n))

            buffer.add(states[i], self.actions[i], n_reward, next_state, done, self.graph)
