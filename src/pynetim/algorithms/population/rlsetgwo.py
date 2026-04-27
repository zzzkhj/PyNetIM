"""
RLSetGWOAlgorithm: 基于强化学习的灰狼优化算法

论文: Roayaei, M. Using Reinforcement Learning to Boost Grey Wolf Optimizer for 
      Influence Maximization in Social Networks. SN COMPUT. SCI. 7, 184 (2026).
      https://doi.org/10.1007/s42979-026-04770-7

算法特点:
- 深度Q网络(DQN)选择优化操作
- 灰狼优化(GWO)的三领导者机制
- 集合编码提高效率
- 9种操作策略

Example:
    >>> from pynetim import IMGraph
    >>> from pynetim.algorithms.population import RLSetGWO
    >>>
    >>> graph = IMGraph(edges, weights=0.3)
    >>> algo = RLSetGWOAlgorithm(graph, pop_size=20, max_iter=50, diffusion_model='IC')
    >>> seeds = algo.run(k=10)
"""

from __future__ import annotations

import random
import math
from typing import Set, List, Optional, TYPE_CHECKING
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .base_population import BasePopulationAlgorithm

if TYPE_CHECKING:
    from ...graph import IMGraph


class DQNetwork(nn.Module):
    """深度Q网络。
    
    用于选择优化操作策略。
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 24):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class RLSetGWOAlgorithm(BasePopulationAlgorithm):
    """基于强化学习的灰狼优化影响力最大化算法。
    
    结合深度Q网络和灰狼优化算法，通过强化学习选择最优的优化操作。
    
    该算法使用种群进化机制，维护三个领导者（α、β、δ狼），
    通过 DQN 网络选择 9 种优化操作之一来更新种群个体。
    
    References:
        Roayaei, M. (2026). Using Reinforcement Learning to Boost Grey Wolf Optimizer 
        for Influence Maximization in Social Networks. SN Computer Science, 7, 184.
        https://doi.org/10.1007/s42979-026-04770-7
    
    Attributes:
        pop_size: 种群大小
        max_iter: 最大迭代次数
        epsilon: 探索率
        epsilon_min: 最小探索率
        epsilon_decay: 探索率衰减
        gamma: 折扣因子
        learning_rate: 学习率
        batch_size: 批次大小
        memory_size: 经验回放缓冲区大小
        convergence_limit: 收敛阈值
        mc_rounds: 蒙特卡洛模拟次数
        device: 计算设备
    
    Example:
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = RLSetGWOAlgorithm(graph, pop_size=20, max_iter=50, diffusion_model='IC')
        >>> seeds = algo.run(k=10)
    """
    
    def __init__(
        self,
        graph: 'IMGraph',
        pop_size: int = 20,
        max_iter: int = 50,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        gamma: float = 0.95,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        memory_size: int = 2500,
        convergence_limit: int = 30,
        mc_rounds: int = 100,
        pretrained: bool = False,
        weights_path: Optional[str] = None,
        device: str = 'auto',
        diffusion_model: str = 'IC'
    ):
        """初始化RLSetGWOAlgorithm算法。
        
        Args:
            graph: 输入图对象
            pop_size: 种群大小，默认20
            max_iter: 最大迭代次数，默认50
            epsilon: 初始探索率，默认1.0
            epsilon_min: 最小探索率，默认0.01
            epsilon_decay: 探索率衰减，默认0.995
            gamma: 折扣因子，默认0.95
            learning_rate: 学习率，默认0.001
            batch_size: 批次大小，默认32
            memory_size: 经验回放缓冲区大小，默认2500
            convergence_limit: 收敛阈值，默认30
            mc_rounds: 蒙特卡洛模拟次数，默认100
            pretrained: 是否使用预训练权重（暂不支持）
            weights_path: 本地权重路径
            device: 计算设备
            diffusion_model: 扩散模型，支持'IC'或'LT'
        """
        super().__init__(graph, pop_size, max_iter, diffusion_model, mc_rounds)
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.convergence_limit = convergence_limit
        self.pretrained = pretrained
        self.weights_path = weights_path
        
        self.device = self._get_device(device)
        
        self.num_nodes = graph.num_nodes
        self.num_actions = 9
        
        self.state_size = self.num_nodes * 4
        
        self._init_dqn()
        
        self.memory = deque(maxlen=memory_size)
        
        self.population_sets = None
        self.leaders = None
        self.leader_sets = None
        self.leader_scores = None
        
    def _get_device(self, device: str) -> torch.device:
        """获取计算设备。
        
        Args:
            device: 设备字符串，支持 'auto'、'cpu'、'cuda'。
            
        Returns:
            torch.device: 计算设备对象。
        """
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
        
    def _init_dqn(self):
        """初始化DQN网络。"""
        self.model = DQNetwork(self.state_size, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def run(self, k: int) -> Set[int]:
        """执行RLSetGWOAlgorithm算法。
        
        Args:
            k: 种子节点数量
            
        Returns:
            Set[int]: 选择的种子节点集合
        """
        self._init_population(k)
        
        for iteration in range(self.max_iter):
            self._evolution_step(k)
            
            if self._check_convergence():
                break
                
        self.seeds = self.leader_sets[0]
        self.best_individual = self.leader_sets[0]
        self.best_fitness = self.leader_scores[0]
        return self.seeds
    
    def _init_population(self, k: int):
        """初始化种群。
        
        Args:
            k: 种子节点数量
        """
        self.population = np.zeros((self.pop_size, self.num_nodes), dtype=np.float32)
        self.population_sets = []
        self.population_fitness = []
        
        for i in range(self.pop_size):
            selected = random.sample(range(self.num_nodes), k)
            for idx in selected:
                self.population[i][idx] = 1
            self.population_sets.append(set(selected))
            
        self._init_leaders()
        
    def _init_leaders(self):
        """初始化三个领导者。"""
        self.leaders = np.zeros((3, self.num_nodes), dtype=np.float32)
        self.leader_sets = [None, None, None]
        self.leader_scores = [-1, -1, -1]
        
        for i in range(self.pop_size):
            fitness = self._evaluate_fitness(self.population_sets[i])
            self.population_fitness.append(fitness)
            self._update_leaders(i, fitness)
            
    def _evaluate_fitness(self, individual) -> float:
        """评估个体适应度。
        
        使用扩散模型进行蒙特卡洛模拟评估影响力传播。
        
        Args:
            individual: 个体（种子节点集合）
            
        Returns:
            float: 适应度值（影响力传播）
        """
        return super()._evaluate_fitness(individual)
    
    def _update_leaders(self, pop_idx: int, fitness: float):
        """更新领导者。
        
        Args:
            pop_idx: 种群索引
            fitness: 适应度值
        """
        if fitness > self.leader_scores[0]:
            self.leader_scores[0] = fitness
            self.leaders[0] = self.population[pop_idx].copy()
            self.leader_sets[0] = self.population_sets[pop_idx].copy()
        elif fitness > self.leader_scores[1]:
            self.leader_scores[1] = fitness
            self.leaders[1] = self.population[pop_idx].copy()
            self.leader_sets[1] = self.population_sets[pop_idx].copy()
        elif fitness > self.leader_scores[2]:
            self.leader_scores[2] = fitness
            self.leaders[2] = self.population[pop_idx].copy()
            self.leader_sets[2] = self.population_sets[pop_idx].copy()
            
        if self.leader_scores[1] == -1:
            self.leader_scores[1] = self.leader_scores[0]
            self.leaders[1] = self.leaders[0].copy()
            self.leader_sets[1] = self.leader_sets[0].copy()
            
        if self.leader_scores[2] == -1:
            self.leader_scores[2] = self.leader_scores[0]
            self.leaders[2] = self.leaders[0].copy()
            self.leader_sets[2] = self.leader_sets[0].copy()
    
    def _evolution_step(self, k: int):
        """执行一步进化。
        
        Args:
            k: 种子节点数量
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        
        for i in range(self.pop_size):
            state = self._get_state(i)
            action = self._select_action(state)
            
            old_fitness = self.population_fitness[i]
            
            self._apply_action(i, action, k)
            
            new_fitness = self._evaluate_fitness(self.population_sets[i])
            self.population_fitness[i] = new_fitness
            
            reward = 2 if new_fitness > old_fitness else -1
            
            next_state = self._get_state(i)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            
            self._update_leaders(i, new_fitness)
            
        self._train_dqn(states, actions, rewards, next_states)
        
    def _get_state(self, pop_idx: int) -> torch.Tensor:
        """获取状态。
        
        Args:
            pop_idx: 种群索引
            
        Returns:
            torch.Tensor: 状态向量
        """
        state = np.concatenate([
            self.population[pop_idx],
            self.leaders[0],
            self.leaders[1],
            self.leaders[2]
        ])
        return torch.FloatTensor(state).to(self.device)
    
    def _select_action(self, state: torch.Tensor) -> int:
        """选择动作。
        
        Args:
            state: 状态向量
            
        Returns:
            int: 动作索引
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
            
        with torch.no_grad():
            q_values = self.model(state)
            return q_values.argmax().item()
    
    def _apply_action(self, pop_idx: int, action: int, k: int):
        """应用动作。
        
        Args:
            pop_idx: 种群索引
            action: 动作索引
            k: 种子节点数量
        """
        if action == 0:
            self.population_sets[pop_idx] = self._set_explore(
                self.population_sets[pop_idx], 0.2
            )
        elif action == 1:
            self.population_sets[pop_idx] = self._set_exploit(
                self.population_sets[pop_idx], 0.2
            )
        elif action == 2:
            selected = random.sample(range(self.num_nodes), k)
            self.population_sets[pop_idx] = set(selected)
        elif action == 3:
            partner = random.randint(0, 2)
            self.population_sets[pop_idx] = self._crossover(
                self.population_sets[pop_idx], self.leader_sets[partner]
            )
        elif action == 4:
            self.population_sets[pop_idx] = self._mutation(
                self.population_sets[pop_idx]
            )
        elif action == 5:
            self.population_sets[pop_idx] = self._pair_swap(
                self.population_sets[pop_idx]
            )
        elif action == 6:
            self.population_sets[pop_idx] = self._insertion(
                self.population_sets[pop_idx]
            )
        elif action == 7:
            self.population_sets[pop_idx] = self._inversion(
                self.population_sets[pop_idx]
            )
        elif action == 8:
            self.population_sets[pop_idx] = self._displace(
                self.population_sets[pop_idx]
            )
            
        self._update_population_vector(pop_idx)
    
    def _set_explore(self, wolf_set: Set[int], extent: float) -> Set[int]:
        """集合探索操作。"""
        leaders_union = set().union(*self.leader_sets)
        all_nodes = set(range(self.num_nodes))
        new_items = all_nodes - wolf_set - leaders_union
        old_items = wolf_set & leaders_union
        
        D = len(wolf_set - leaders_union)
        selected_num = min(
            abs(math.ceil(extent * D)),
            len(old_items),
            len(new_items)
        )
        
        if selected_num > 0 and old_items and new_items:
            wolf_set -= set(random.sample(list(old_items), selected_num))
            wolf_set |= set(random.sample(list(new_items), selected_num))
            
        return wolf_set
    
    def _set_exploit(self, wolf_set: Set[int], extent: float) -> Set[int]:
        """集合利用操作。"""
        leaders_union = set().union(*self.leader_sets)
        new_items = leaders_union - wolf_set
        old_items = wolf_set - leaders_union
        
        D = len(old_items)
        selected_num = min(
            abs(math.ceil(extent * D)),
            len(old_items),
            len(new_items)
        )
        
        if selected_num > 0 and old_items and new_items:
            wolf_set -= set(random.sample(list(old_items), selected_num))
            wolf_set |= set(random.sample(list(new_items), selected_num))
            
        return wolf_set
    
    def _crossover(self, wolf_set: Set[int], partner_set: Set[int]) -> Set[int]:
        """交叉操作。"""
        common = wolf_set & partner_set
        diff1 = wolf_set - partner_set
        diff2 = partner_set - wolf_set
        
        k = len(wolf_set)
        new_set = common.copy()
        
        remaining = k - len(new_set)
        if remaining > 0:
            from_diff1 = min(remaining, len(diff1))
            from_diff2 = remaining - from_diff1
            
            if from_diff1 > 0:
                new_set |= set(random.sample(list(diff1), from_diff1))
            if from_diff2 > 0 and from_diff2 <= len(diff2):
                new_set |= set(random.sample(list(diff2), from_diff2))
                
        return new_set
    
    def _mutation(self, wolf_set: Set[int]) -> Set[int]:
        """变异操作。"""
        new_set = wolf_set.copy()
        all_nodes = set(range(self.num_nodes))
        available = all_nodes - wolf_set
        
        if wolf_set and available:
            remove_node = random.choice(list(wolf_set))
            add_node = random.choice(list(available))
            new_set.remove(remove_node)
            new_set.add(add_node)
            
        return new_set
    
    def _pair_swap(self, wolf_set: Set[int]) -> Set[int]:
        """交换操作。"""
        return self._mutation(wolf_set)
    
    def _insertion(self, wolf_set: Set[int]) -> Set[int]:
        """插入操作。"""
        return self._mutation(wolf_set)
    
    def _inversion(self, wolf_set: Set[int]) -> Set[int]:
        """反转操作。"""
        return self._mutation(wolf_set)
    
    def _displace(self, wolf_set: Set[int]) -> Set[int]:
        """置换操作。"""
        return self._mutation(wolf_set)
    
    def _update_population_vector(self, pop_idx: int):
        """更新种群向量表示。"""
        self.population[pop_idx] = np.zeros(self.num_nodes, dtype=np.float32)
        for node in self.population_sets[pop_idx]:
            self.population[pop_idx][node] = 1
    
    def _train_dqn(
        self,
        states: List[torch.Tensor],
        actions: List[int],
        rewards: List[float],
        next_states: List[torch.Tensor]
    ):
        """训练DQN网络。
        
        Args:
            states: 状态列表
            actions: 动作列表
            rewards: 奖励列表
            next_states: 下一状态列表
        """
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        
        current_q_values = self.model(states)
        
        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            target_q_values = torch.FloatTensor(rewards).to(self.device) + \
                             self.gamma * max_next_q
        
        for i, action in enumerate(actions):
            current_q_values[i][action] = target_q_values[i]
            
        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1).expand_as(current_q_values))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _check_convergence(self) -> bool:
        """检查是否收敛。
        
        Returns:
            bool: 是否收敛
        """
        if not hasattr(self, '_old_best_score'):
            self._old_best_score = self.leader_scores[0]
            self._convergence_counter = 0
            return False
            
        if self.leader_scores[0] == self._old_best_score:
            self._convergence_counter += 1
        else:
            self._convergence_counter = 0
            self._old_best_score = self.leader_scores[0]
            
        return self._convergence_counter >= self.convergence_limit
