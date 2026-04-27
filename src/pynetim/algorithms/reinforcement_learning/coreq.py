from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

from ...graph import IMGraph, compute_k_shell_values
from .base_rl import BaseRLAlgorithm

if TYPE_CHECKING:
    from ...graph import IMGraph


class CoreQAlgorithm(BaseRLAlgorithm):
    """K-core 层次引导的 Q-learning 影响力最大化算法。
    
    结合 K-core 分解和强化学习进行种子节点选择，
    使用最大似然估计方法选择候选种子集。
    
    算法框架：
        1. K-core 分解：识别网络核心层次结构
        2. 候选集生成：基于最大似然估计筛选候选种子
        3. Q-learning 优化：通过强化学习迭代优化种子选择策略
    
    与 TCQAlgorithm 的区别：
        - CoreQ: 传统 IM，最大似然候选选择
        - TCQ: Targeted IM，目标节点筛选
    
    References:
        Ahmad, W., & Wang, B. (2025). A learning-based influence maximization 
        framework for complex networks via K-core hierarchies and reinforcement 
        learning. Expert Systems with Applications, 259, 125393.
    
    Attributes:
        n_candidates: 候选集大小
        episodes: Q-learning 训练轮数
    
    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms.reinforcement_learning import CoreQAlgorithm
        >>> 
        >>> # 创建图
        >>> edges = [(0, 1, 0.1), (0, 2, 0.15), (1, 2, 0.2)]
        >>> graph = IMGraph(edges, directed=True)
        >>> 
        >>> # 初始化算法
        >>> algo = CoreQAlgorithm(
        ...     graph,
        ...     n_candidates=60,   # 候选集大小
        ...     episodes=200,      # 训练轮数
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
        n_candidates: int = 60,
        episodes: int = 200,
        random_seed: Optional[int] = None,
        **kwargs
    ):
        """初始化 CoreQ 算法。
        
        Args:
            graph: IMGraph 图对象。
            n_candidates: 候选集大小，默认 60。
            episodes: Q-learning 训练轮数，默认 200。
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
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self._k_core: Dict[int, int] = {}
        self._candidate_set: List[int] = []
        self.q_table: Optional[np.ndarray] = None

    def _compute_k_core(self) -> Dict[int, int]:
        """计算 K-shell 分解。
        
        Returns:
            Dict[int, int]: 节点到 K-shell 值的映射。
        """
        return compute_k_shell_values(self.graph)

    def _compute_likelihood_score(self, node: int) -> float:
        """计算节点的最大似然得分。
        
        基于论文公式：Sc(w) = Σ(1 - p_uw)
        其中 p_uw 是邻居 u 激活节点 w 的概率。
        
        得分越高，节点越有可能影响整个网络。
        
        Args:
            node: 节点 ID。
            
        Returns:
            float: 最大似然得分。
        """
        in_neighbors = self.graph.in_neighbors(node)
        if not in_neighbors:
            return 0.0
        
        score = 0.0
        for u in in_neighbors:
            p_uw = self.graph.get_edge_weight(u, node)
            score += (1 - p_uw)
        
        return score

    def _get_candidate_seeds(self) -> List[int]:
        """基于最大似然估计选择候选种子集。
        
        计算每个节点的似然得分，选择得分最高的 n_candidates 个节点。
        
        Returns:
            List[int]: 候选种子节点列表。
        """
        nodes = list(range(self.graph.num_nodes))
        
        node_scores = []
        for node in nodes:
            score = self._compute_likelihood_score(node)
            k_core = self._k_core.get(node, 0)
            node_scores.append((node, score, k_core))
        
        node_scores.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        candidates = [node for node, _, _ in node_scores[:self.n_candidates]]
        
        return candidates

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

    def _compute_activation_probability(self, node: int, seed_set: Set[int]) -> float:
        """计算节点被激活的概率。
        
        P(w) = 1 - (1 - p_vw)^r
        其中 p_vw 是边权重，r 是种子集中邻居的数量。
        
        Args:
            node: 目标节点。
            seed_set: 种子节点集合。
            
        Returns:
            float: 激活概率。
        """
        neighbors = set(self.graph.in_neighbors(node))
        seed_neighbors = neighbors & seed_set
        
        if not seed_neighbors:
            return 0.0
        
        prob_not_activated = 1.0
        for u in seed_neighbors:
            p_uw = self.graph.get_edge_weight(u, node)
            prob_not_activated *= (1 - p_uw)
        
        return 1 - prob_not_activated

    def _compute_sigma(self, seed_set: Set[int]) -> float:
        """计算种子集的影响力传播（近似）。
        
        使用代理函数近似计算影响力传播，避免蒙特卡洛模拟。
        
        Args:
            seed_set: 种子节点集合。
            
        Returns:
            float: 影响力传播值。
        """
        gamma_s = set()
        for node in seed_set:
            gamma_s.update(self.graph.out_neighbors(node))
        
        gamma_s = gamma_s - seed_set
        
        sigma = len(seed_set) + sum(self._compute_activation_probability(node, seed_set) 
                                    for node in gamma_s)
        
        return sigma

    def _argmax_q(self, q_table: np.ndarray, state: int, visited: Set[int], 
                  candidate_set: List[int]) -> Optional[int]:
        """选择 Q 值最大的动作。
        
        Args:
            q_table: Q 表。
            state: 当前状态。
            visited: 已访问的节点集合。
            candidate_set: 候选节点列表。
            
        Returns:
            Optional[int]: 选择的动作索引，如果没有可用动作则返回 None。
        """
        max_value = float('-inf')
        best_action = None
        
        for idx in range(len(candidate_set)):
            if candidate_set[idx] not in visited:
                if q_table[state, idx] > max_value:
                    max_value = q_table[state, idx]
                    best_action = idx
        
        return best_action

    def _max_q(self, q_table: np.ndarray, action: int, visited: Set[int], 
               candidate_set: List[int]) -> float:
        """获取下一状态的最大 Q 值。
        
        Args:
            q_table: Q 表。
            action: 当前动作。
            visited: 已访问的节点集合。
            candidate_set: 候选节点列表。
            
        Returns:
            float: 最大 Q 值。
        """
        max_value = float('-inf')
        for idx, value in enumerate(q_table[action, :]):
            if candidate_set[idx] not in visited and value > max_value:
                max_value = value
        return max_value

    def run(self, k: int, **kwargs) -> Set[int]:
        """运行 CoreQ 算法。
        
        Args:
            k: 种子节点数量。
            
        Returns:
            Set[int]: 选中的种子节点集合。
        """
        self._k_core = self._compute_k_core()
        self._candidate_set = self._get_candidate_seeds()
        
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
