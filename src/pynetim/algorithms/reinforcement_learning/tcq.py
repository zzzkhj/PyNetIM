"""
TCQ: Targeted Core-based Q-learning framework

K-core-Guided Adaptive Learning and Policy Optimization for 
Targeted Influence Maximization in Complex Networks

Copyright (c) 2025 Waseem Ahmad, Bang Wang

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Original repository: https://github.com/User2021-ai/TCQ

Paper: K-core-Guided Adaptive Learning and Policy Optimization for 
       Targeted Influence Maximization in Complex Networks
Journal: Neurocomputing, 2025
DOI: https://doi.org/10.1016/j.neucom.2025.131612

Modified by: PyNetIM Team for integration into PyNetIM
"""

from __future__ import annotations

import heapq
import random
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

from ...graph import IMGraph, compute_k_shell_values
from .base_rl import BaseRLAlgorithm

if TYPE_CHECKING:
    from ...graph import IMGraph


class TCQAlgorithm(BaseRLAlgorithm):
    """K-core 引导的自适应 Q-learning 影响力最大化算法。
    
    结合 K-core 分解和强化学习进行种子节点选择。
    
    算法框架：
        1. K-core 分解：识别网络核心层次结构
        2. 候选集生成：基于 K-core 和目标节点筛选候选种子
        3. Q-learning 优化：通过强化学习迭代优化种子选择策略
    
    References:
        Ahmad, W., & Wang, B. (2025). K-core-guided adaptive learning 
        and policy optimization for targeted influence maximization 
        in complex networks. Neurocomputing, 131612.
    
    Attributes:
        n_candidates: 候选集倍数，候选集大小 = n_candidates * k
        episodes: Q-learning 训练轮数
        target_labels: 目标节点标签集合（可选）
        node_labels: 节点标签字典（可选）
    
    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms.reinforcement_learning import TCQAlgorithm
        >>> 
        >>> # 创建图
        >>> edges = [(0, 1, 0.1), (0, 2, 0.15), (1, 2, 0.2)]
        >>> graph = IMGraph(edges, directed=True)
        >>> 
        >>> # 初始化算法
        >>> algo = TCQAlgorithm(
        ...     graph,
        ...     n_candidates=6,   # 候选集倍数
        ...     episodes=200,     # 训练轮数
        ...     random_seed=42
        ... )
        >>> 
        >>> # 运行算法
        >>> seeds = algo.run(k=10)
        >>> print(f"种子节点: {seeds}")
    """

    def __init__(
        self,
        graph: IMGraph,
        n_candidates: int = 6,
        episodes: int = 200,
        target_labels: Optional[Set[str]] = None,
        node_labels: Optional[Dict[int, str]] = None,
        random_seed: Optional[int] = None,
        **kwargs
    ):
        """初始化 TCQ 算法。
        
        Args:
            graph: IMGraph 图对象。
            n_candidates: 候选集倍数，候选集大小 = n_candidates * k，默认 6。
            episodes: Q-learning 训练轮数，默认 200。
            target_labels: 目标节点标签集合，如果为 None 则所有节点都是目标。
            node_labels: 节点标签字典 {node_id: label}，用于目标约束 IM。
            random_seed: 随机种子，用于结果复现。
            **kwargs: 传递给父类的参数，包括：
                - alpha: 学习率，默认 0.1
                - gamma: 折扣因子，默认 0.9
                - epsilon: 探索率，默认 0.5
                - diffusion_model: 扩散模型，默认 'IC'
                - mc_rounds: 蒙特卡洛模拟次数，默认 100
        """
        super().__init__(graph, **kwargs)
        
        self.n_candidates = n_candidates
        self.episodes = episodes
        self.target_labels = target_labels
        self.node_labels = node_labels or {}
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self._k_core = None
        self._candidate_set = None
        self._l_values = None

    def _compute_k_core(self) -> Dict[int, int]:
        """计算 K-shell 分解。
        
        Returns:
            Dict[int, int]: 节点到 K-shell 值的映射。
        """
        return compute_k_shell_values(self.graph)

    def _get_candidate_seeds(self, k: int) -> List[int]:
        """获取候选种子集。
        
        基于 K-core 和目标节点筛选候选种子。
        
        Args:
            k: 种子节点数量。
            
        Returns:
            List[int]: 候选种子节点列表。
        """
        k_core = self._k_core
        degree_dict = {node: self.graph.degree(node) for node in range(self.graph.num_nodes)}
        
        target_set = self.target_labels
        l_values = {}
        
        for node in range(self.graph.num_nodes):
            neighbors = self.graph.out_neighbors(node)
            neighbors_smaller = [n for n in neighbors if k_core[n] <= k_core[node]]
            
            if target_set is not None and self.node_labels:
                count = sum(1 for n in neighbors_smaller 
                           if self.node_labels.get(n) in target_set)
            else:
                count = len(neighbors_smaller)
            
            degree = degree_dict[node]
            l_values[node] = count * (1 - 1 / degree) if neighbors_smaller and degree else -1
        
        top_k = self.n_candidates * k + 1 if self.n_candidates == 1 else self.n_candidates * k
        top_nodes = heapq.nlargest(top_k, l_values, key=l_values.get)
        
        self._l_values = l_values
        return top_nodes

    def _compute_activation_probability(self, node: int, seed_set: Set[int]) -> float:
        """计算节点被激活的概率。
        
        P(w) = 1 - (1 - 1/d)^r
        其中 d 是节点度数，r 是种子集中邻居的数量。
        
        Args:
            node: 目标节点。
            seed_set: 种子节点集合。
            
        Returns:
            float: 激活概率。
        """
        if self.target_labels is not None and self.node_labels:
            if self.node_labels.get(node) not in self.target_labels:
                return 0.0
        
        neighbors = set(self.graph.out_neighbors(node))
        r_v = len(neighbors & seed_set)
        degree = self.graph.degree(node)
        
        return (1 - (1 - 1 / degree) ** r_v) if degree else 0.0

    def _compute_sigma(self, seed_set: Set[int]) -> float:
        """计算种子集的影响力传播。
        
        Args:
            seed_set: 种子节点集合。
            
        Returns:
            float: 影响力传播值。
        """
        gamma_s = set()
        for node in seed_set:
            gamma_s.update(self.graph.out_neighbors(node))
        
        gamma_s = gamma_s - seed_set
        
        sigma = sum(self._compute_activation_probability(node, seed_set) 
                   for node in gamma_s)
        
        return sigma

    def _argmax_q(self, q_table: np.ndarray, state: int, visited: Set[int], 
                  candidate_set: List[int]) -> Optional[int]:
        """选择 Q 值最大的动作。
        
        Args:
            q_table: Q 表。
            state: 当前状态。
            visited: 已访问的节点集合。
            candidate_set: 候选集。
            
        Returns:
            Optional[int]: 选择的动作索引，如果没有有效动作则返回 None。
        """
        visited_indices = {candidate_set.index(node) for node in visited if node in candidate_set}
        valid_actions = [i for i in range(q_table.shape[1]) if i not in visited_indices]
        
        if not valid_actions:
            return None
        
        return valid_actions[np.argmax(q_table[state, valid_actions])]

    def _max_q(self, q_table: np.ndarray, action: int, visited: Set[int], 
               candidate_set: List[int]) -> float:
        """获取最大 Q 值。
        
        Args:
            q_table: Q 表。
            action: 当前动作。
            visited: 已访问的节点集合。
            candidate_set: 候选集。
            
        Returns:
            float: 最大 Q 值。
        """
        max_value = float('-inf')
        for idx, value in enumerate(q_table[action, :]):
            if candidate_set[idx] not in visited and value > max_value:
                max_value = value
        return max_value

    def _init_q_table(self) -> np.ndarray:
        """初始化 Q 表。
        
        Returns:
            np.ndarray: 初始化的 Q 表。
        """
        m = len(self._candidate_set)
        q_table = np.zeros([m, m])
        
        for j in range(m):
            q_value = self._compute_sigma({self._candidate_set[j], self._candidate_set[0]})
            q_table[:, j] = q_value
        
        return q_table

    def run(self, k: int, **kwargs) -> Set[int]:
        """运行 TCQ 算法。
        
        Args:
            k: 种子节点数量。
            
        Returns:
            Set[int]: 选中的种子节点集合。
        """
        self._k_core = self._compute_k_core()
        self._candidate_set = self._get_candidate_seeds(k)
        
        if len(self._candidate_set) < k:
            self._candidate_set = list(range(self.graph.num_nodes))
        
        initial_state = 0
        m = len(self._candidate_set)
        
        q_table = self._init_q_table()
        
        for episode in range(self.episodes):
            state = initial_state
            visited = {self._candidate_set[state]}
            prev_sigma = 0
            
            for _ in range(k - 1):
                if random.random() < self.epsilon:
                    possible_actions = [i for i in range(m) 
                                       if self._candidate_set[i] not in visited]
                    if not possible_actions:
                        break
                    action = random.choice(possible_actions)
                else:
                    action = self._argmax_q(q_table, state, visited, self._candidate_set)
                    if action is None:
                        break
                
                current_sigma = self._compute_sigma(visited | {self._candidate_set[action]})
                reward = current_sigma - prev_sigma
                prev_sigma = current_sigma
                
                max_q = self._max_q(q_table, action, visited | {self._candidate_set[action]}, 
                                   self._candidate_set)
                q_table[state, action] += self.alpha * (reward + self.gamma * max_q - q_table[state, action])
                
                state = action
                visited.add(self._candidate_set[state])
        
        selected_nodes = []
        state = initial_state
        selected_nodes.append(self._candidate_set[state])
        
        while len(selected_nodes) < k:
            action = self._argmax_q(q_table, state, set(selected_nodes), self._candidate_set)
            if action is None:
                break
            state = action
            selected_nodes.append(self._candidate_set[state])
        
        self.seeds = set(selected_nodes[:k])
        return self.seeds
