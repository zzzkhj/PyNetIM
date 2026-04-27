"""
SADPEA: Structure-aware Dual Probability Evolutionary Adaptive Algorithm

结构感知双概率进化自适应算法。
"""

import copy
import math
import random
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

from ...graph import IMGraph
from .base_population import BasePopulationAlgorithm


class SADPEAAlgorithm(BasePopulationAlgorithm):
    """结构感知双概率进化自适应算法
    
    两阶段算法框架：
        1. 进化算法阶段：双概率池的变异和交叉
        2. 模拟退火阶段：自适应温度调整
    
    这是预算约束的影响力最大化算法，内部自动计算预算。
    
    References:
        Zhu, E., Wang, H., Zhang, Y., & Ma, M. (2025). SADPEA: Structure-aware 
        dual probability evolutionary adaptive algorithm for the budgeted influence 
        maximization problem. Information Sciences, 699, 121784.
    
    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms.population import SADPEA
        >>> 
        >>> # 创建图
        >>> edges = [(0, 1, 0.1), (0, 2, 0.15), (1, 2, 0.2)]
        >>> graph = IMGraph(edges, directed=True)
        >>> 
        >>> # 初始化算法
        >>> algo = SADPEAAlgorithm(graph, pop_size=20, max_iter=50)
        >>> 
        >>> # 运行算法
        >>> seeds = algo.run(k=10)
    """
    
    def __init__(
        self,
        graph: IMGraph,
        pop_size: int = 30,
        max_iter: int = 140,
        diffusion_model: str = 'IC',
        mc_rounds: int = 100,
        dp: float = 0.6,
        mp_1: float = 0.2,
        mp_2: float = 0.1,
        cp: float = 0.6,
        sa_initial_temp: float = 2000,
        sa_final_temp: float = 20,
        sa_cooling_coeff: int = 5,
        sa_iterations_per_temp: int = 20,
        node_costs: Optional[Dict[int, float]] = None,
        random_seed: Optional[int] = None
    ):
        """初始化SADPEAAlgorithm算法
        
        Args:
            graph: IMGraph图对象
            pop_size: 种群大小，默认为30
            max_iter: 最大进化代数，默认为140
            diffusion_model: 扩散模型名称，支持 'IC' 或 'LT'，默认为 'IC'
            mc_rounds: 蒙特卡洛模拟次数，默认为100
            dp: 初始化扰动概率，默认为0.6
            mp_1: 高成本节点变异概率，默认为0.2
            mp_2: 低成本节点变异概率，默认为0.1
            cp: 交叉概率，默认为0.6
            sa_initial_temp: 模拟退火初始温度，默认为2000
            sa_final_temp: 模拟退火终止温度，默认为20
            sa_cooling_coeff: 模拟退火冷却系数，默认为5
            sa_iterations_per_temp: 每个温度的迭代次数，默认为20
            node_costs: 节点成本字典，如果为None则使用度中心性计算
            random_seed: 随机种子，用于结果复现
        """
        super().__init__(graph, pop_size, max_iter, diffusion_model, mc_rounds)
        
        self.dp = dp
        self.mp_1 = mp_1
        self.mp_2 = mp_2
        self.cp = cp
        self.sa_initial_temp = sa_initial_temp
        self.sa_final_temp = sa_final_temp
        self.sa_cooling_coeff = sa_cooling_coeff
        self.sa_iterations_per_temp = sa_iterations_per_temp
        
        # 设置随机种子
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # 节点数量
        self.num_nodes = graph.num_nodes
        
        # 节点成本
        self.node_costs = self._init_node_costs(node_costs)
        
        # 节点影响力评估（用于初始化和排序）
        self.node_influence = self._calculate_node_influence()
        
        # 种群
        self.population: List[List[int]] = []
        
        # 候选池（按影响力排序的节点列表）
        self.pool_list = self._create_candidate_pool()
    
    def _init_node_costs(self, node_costs: Optional[Dict[int, float]]) -> Dict[int, float]:
        """初始化节点成本
        
        如果没有提供成本，则使用度中心性计算。
        计算公式：node_cost = 1 + degree * 0.01
        
        Args:
            node_costs: 自定义节点成本字典，如果为None则自动计算
            
        Returns:
            节点成本字典，键为节点ID，值为成本
        """
        if node_costs is not None:
            return node_costs
        
        costs = {}
        for node in range(self.num_nodes):
            degree = self.graph.out_degree(node)
            costs[node] = 1 + degree * 0.01
        
        return costs
    
    def _calculate_node_influence(self) -> Dict[int, float]:
        """计算节点影响力
        
        使用两跳混合度分解（Two-hop Mixed Degree Decomposition, MDD）。
        
        Returns:
            节点影响力字典，键为节点ID，值为MDD影响力值
        """
        node_dict = {}
        mdd_dict = {}
        
        # 阶段1：计算初始MDD值
        average = 0.09995
        variance = 0.01
        
        for node in range(self.num_nodes):
            # 随机激活概率
            activate_weight = np.random.normal(average, variance)
            while activate_weight < 0.001 or activate_weight > 0.2:
                activate_weight = np.random.normal(average, variance)
            
            # 计算两跳邻居影响
            node_index = 0
            neighbors = self.graph.out_neighbors(node)
            for neighbor in neighbors:
                node_index += 1
                neighbor_degree = self.graph.out_degree(neighbor)
                node_index += activate_weight * neighbor_degree
            
            node_dict[node] = [node_index, set(neighbors)]
        
        # 阶段2：迭代分解
        temp_nodes = set(range(self.num_nodes))
        rank = 1
        factor = 0.7  # 原始代码的参数
        
        while len(mdd_dict) < self.num_nodes:
            little_node_list = []
            little_node_neighbor = set()
            
            for key, value in list(node_dict.items()):
                if value[0] <= rank:
                    mdd_dict[key] = value[0]
                    little_node_list.append(key)
                    little_node_neighbor = little_node_neighbor.union(value[1])
            
            little_node_neighbor = little_node_neighbor.difference(little_node_list)
            
            for del_node in little_node_list:
                if del_node in node_dict:
                    node_dict.pop(del_node)
                if del_node in temp_nodes:
                    temp_nodes.remove(del_node)
            
            if len(little_node_list) > 0:
                for new_node in little_node_neighbor:
                    if new_node not in temp_nodes or new_node not in node_dict:
                        continue
                    
                    activate_weight = np.random.normal(average, variance)
                    while activate_weight < 0.001 or activate_weight > 0.2:
                        activate_weight = np.random.normal(average, variance)
                    
                    mdd_influence_spread = 0
                    neighbors = self.graph.out_neighbors(new_node)
                    for neighbor in neighbors:
                        if neighbor in temp_nodes:
                            mdd_influence_spread += 1
                            neighbor_degree = self.graph.out_degree(neighbor)
                            mdd_influence_spread += activate_weight * neighbor_degree
                        else:
                            mdd_influence_spread += factor
                            neighbor_degree = self.graph.out_degree(neighbor)
                            mdd_influence_spread += factor * activate_weight * neighbor_degree
                    
                    node_dict[new_node] = [mdd_influence_spread, 
                                          set(neighbors).intersection(temp_nodes)]
                
                little_node_list.clear()
                little_node_neighbor.clear()
            else:
                rank += 0.5
        
        return mdd_dict
    
    def _create_candidate_pool(self) -> List[int]:
        """创建候选池
        
        按影响力/成本比排序
        """
        mdd_list = [(node, influence / self.node_costs[node]) 
                    for node, influence in self.node_influence.items()]
        mdd_list.sort(key=lambda x: x[1], reverse=True)
        
        return [x[0] for x in mdd_list]
    
    def _fitness_function(self, seed_set: Set[int]) -> float:
        """适应度函数
        
        fitness_function_3
        计算种子集的影响传播（使用随机激活概率）
        """
        influence_spread = 0
        neighbor_dict = {}
        
        average = 0.09995
        variance = 0.01
        
        # 计算直接激活的邻居
        for node in seed_set:
            influence_spread += 1
            neighbors = self.graph.out_neighbors(node)
            for neighbor in neighbors:
                # 获取边权重
                weight = self.graph.get_edge_weight(node, neighbor)
                if neighbor in neighbor_dict:
                    neighbor_dict[neighbor].append(weight)
                else:
                    neighbor_dict[neighbor] = [weight]
        
        # 计算邻居的激活概率
        for key, weights in neighbor_dict.items():
            neighbor_influence_spread = 1
            for weight in weights:
                neighbor_influence_spread *= (1 - weight)
            
            neighbor_influence_spread = 1 - neighbor_influence_spread
            
            # 随机激活权重
            activate_weight = np.random.normal(average, variance)
            while activate_weight < 0.001 or activate_weight > 0.2:
                activate_weight = np.random.normal(average, variance)
            
            out_degree = self.graph.out_degree(key)
            neighbor_influence_spread *= (1 + out_degree * activate_weight)
            influence_spread += neighbor_influence_spread
        
        return influence_spread
    
    def _low_bound_cost(self, pool: List[int]) -> float:
        """计算节点成本下界
        
        low_bound_cost
        """
        budget_rank = [(node, self.node_costs[node]) for node in pool]
        budget_rank.sort(key=lambda x: x[1], reverse=False)
        
        if len(budget_rank) <= 20:
            return budget_rank[0][1]
        else:
            min_budget = 0
            min_len = int(0.05 * len(budget_rank))
            for i in range(min_len):
                min_budget += budget_rank[i][1]
            
            return min_budget / min_len
    
    def _initialize_population(self, budget: float) -> List[List[int]]:
        """初始化种群
        
        初始化逻辑
        """
        n = self.num_nodes
        k = max(1, int(budget / 2))  # 初始种子集大小，至少为1
        
        initial_pop = []
        
        for _ in range(self.pop_size):
            # 随机范围计算（遵循原始公式）
            p = random.uniform(0.1, 0.5)
            random_range = k + n * math.pow(budget / (2 * (n - k)), 1 - p) * math.sin(math.pi * p / 2)
            random_range = max(k, min(int(random_range), len(self.pool_list)))  # 确保范围有效
            random_range_list = self.pool_list[:random_range]
            
            # 初始个体
            initial_individuals = list(self.pool_list[:k])
            
            # 扰动
            if len(random_range_list) > 0:
                for j in range(min(k, len(initial_individuals))):
                    p_div = random.random()
                    if p_div < self.dp:
                        # 找到不在 initial_individuals 中的候选节点
                        candidates = [node for node in random_range_list if node not in initial_individuals]
                        if len(candidates) > 0:
                            select_node = random.choice(candidates)
                            initial_individuals[j] = select_node
            
            initial_pop.append(initial_individuals)
        
        return initial_pop
    
    def _evolution_stage(self, budget: float) -> Set[int]:
        """进化算法阶段
        
        two_stage_evolution_sa的第一阶段。
        执行双概率变异、交叉和选择操作。
        
        Args:
            budget: 预算约束
            
        Returns:
            第二概率池的节点集合，用于模拟退火阶段
        """
        n = self.num_nodes
        k = int(budget / 2)
        
        # 初始化种群
        self.population = self._initialize_population(budget)
        
        # 进化迭代
        for _ in range(self.max_iter):
            # 双概率变异
            mutation_pop = self._dual_probability_mutation(budget, k)
            
            # 交叉
            crossover_pop = self._dual_probability_crossover(mutation_pop, k)
            
            # 选择
            self.population = self._selection(mutation_pop)
        
        # 收集所有个体中的节点作为候选池
        pool_2_set = set()
        for individual in self.population:
            pool_2_set = pool_2_set.union(set(individual))
        
        return pool_2_set
    
    def _dual_probability_mutation(self, budget: float, k: int) -> List[List[int]]:
        """双概率变异
        
        高成本节点用mp_1，低成本节点用mp_2。
        这是SADPEAAlgorithm的核心创新之一。
        
        Args:
            budget: 预算约束
            k: 种子节点数量
            
        Returns:
            变异后的种群列表
        """
        n = self.num_nodes
        mutation_pop = []
        
        for i in range(self.pop_size):
            # 随机范围计算
            p = random.uniform(0.1, 0.5)
            random_range = k + n * math.pow(budget / (2 * (n - k)), 1 - p) * math.sin(math.pi * p / 2)
            random_range = max(k, min(int(random_range), len(self.pool_list)))  # 确保范围有效
            random_range_list = list(self.pool_list[:random_range])
            
            # 按成本排序个体
            individuals_budget_rank = [[node, self.node_costs[node]] 
                                      for node in self.population[i]]
            individuals_budget_rank.sort(key=lambda x: x[1], reverse=True)
            mutation_individuals = [x[0] for x in individuals_budget_rank]
            
            # 双概率变异
            if len(random_range_list) > 0:
                for j in range(min(k, len(mutation_individuals))):
                    # 找到不在 mutation_individuals 中的候选节点
                    candidates = [node for node in random_range_list if node not in mutation_individuals]
                    if len(candidates) == 0:
                        break
                    
                    if j < k / 2:
                        # 高成本节点：使用mp_1
                        p_mutation = random.random()
                        if p_mutation < self.mp_1:
                            mutation_node = random.choice(candidates)
                            idx = random_range_list.index(mutation_node)
                            # 交换
                            mutation_individuals[j], random_range_list[idx] = \
                                random_range_list[idx], mutation_individuals[j]
                    else:
                        # 低成本节点：使用mp_2
                        p_mutation = random.random()
                        if p_mutation < self.mp_2:
                            mutation_node = random.choice(candidates)
                            idx = random_range_list.index(mutation_node)
                            # 交换
                            mutation_individuals[j], random_range_list[idx] = \
                                random_range_list[idx], mutation_individuals[j]
            
            mutation_pop.append(mutation_individuals)
        
        return mutation_pop
    
    def _dual_probability_crossover(self, mutation_pop: List[List[int]], k: int) -> List[List[int]]:
        """双概率交叉
        
        交叉逻辑
        """
        n = self.num_nodes
        crossover_pop = []
        
        for i in range(self.pop_size):
            crossover_individuals = []
            
            for j in range(min(k, len(mutation_pop[i]) if i < len(mutation_pop) else 0, 
                              len(self.population[i]) if i < len(self.population) else 0)):
                p_crossover = random.random()
                if p_crossover < self.cp:
                    # 从变异个体中选择
                    if mutation_pop[i][j] not in crossover_individuals:
                        crossover_individuals.append(mutation_pop[i][j])
                    elif self.population[i][j] not in crossover_individuals:
                        crossover_individuals.append(self.population[i][j])
                    else:
                        # 随机选择新节点
                        candidates = [node for node in self.pool_list if node not in crossover_individuals]
                        if len(candidates) > 0:
                            crossover_individuals.append(random.choice(candidates))
                else:
                    # 从原始个体中选择
                    if self.population[i][j] not in crossover_individuals:
                        crossover_individuals.append(self.population[i][j])
                    elif mutation_pop[i][j] not in crossover_individuals:
                        crossover_individuals.append(mutation_pop[i][j])
                    else:
                        candidates = [node for node in self.pool_list if node not in crossover_individuals]
                        if len(candidates) > 0:
                            crossover_individuals.append(random.choice(candidates))
            
            crossover_pop.append(crossover_individuals)
        
        return crossover_pop
    
    def _selection(self, mutation_pop: List[List[int]]) -> List[List[int]]:
        """选择
        
        选择适应度更高的个体
        """
        select_pop = []
        
        for i in range(self.pop_size):
            initial_influence = self._fitness_function(set(self.population[i]))
            mutation_influence = self._fitness_function(set(mutation_pop[i]))
            
            if initial_influence > mutation_influence:
                select_pop.append(self.population[i])
            else:
                select_pop.append(mutation_pop[i])
        
        return select_pop
    
    def _adaptive_simulated_annealing(
        self, 
        budget: float, 
        seed: Set[int], 
        pool: Set[int]
    ) -> List[int]:
        """自适应模拟退火
        
        adaptive_simulated_annealing。
        根据操作的成功率自适应调整温度和操作概率。
        
        Args:
            budget: 预算约束
            seed: 初始种子集合
            pool: 候选节点池
            
        Returns:
            优化后的种子节点列表
        """
        # 初始化参数
        t_f = self.sa_final_temp
        t_t = self.sa_initial_temp
        theta = self.sa_cooling_coeff
        n = self.sa_iterations_per_temp
        
        op1_defeat_num = 0
        op1_success_num = 0
        op2_defeat_num = 0
        op2_success_num = 0
        old_success_num = 0
        op = 0.5
        r = 0
        
        seed = list(seed)
        pool = list(pool)
        
        # 计算当前预算
        current_budget = sum(self.node_costs[node] for node in seed)
        
        # 补充节点以满足预算
        min_node_budget = self._low_bound_cost(pool)
        random.shuffle(pool)
        for node in pool:
            if budget - current_budget < min_node_budget:
                break
            if node not in seed and self.node_costs[node] < budget - current_budget:
                seed.append(node)
                current_budget += self.node_costs[node]
        
        # 自适应温度调整
        while t_t > t_f:
            # 按成本排序种子
            seed_budget_rank = [[node, self.node_costs[node]] for node in seed]
            seed_budget_rank.sort(key=lambda x: x[1], reverse=True)
            seed = [x[0] for x in seed_budget_rank]
            
            seed_spread = self._fitness_function(set(seed))
            
            for _ in range(n):
                p_op = random.random()
                
                if p_op < op:
                    # 操作1：高成本节点
                    # 找到满足条件的节点对
                    found = False
                    for _ in range(100):  # 最多尝试100次
                        node_x = random.randint(0, max(0, int(len(seed) / 2) - 1))
                        node_y = random.randint(0, len(pool) - 1)
                        
                        if (self.node_costs[pool[node_y]] - self.node_costs[seed[node_x]] < budget - current_budget
                            and pool[node_y] not in seed):
                            found = True
                            break
                    
                    if not found:
                        continue
                    
                    # 交换
                    seed[node_x], pool[node_y] = pool[node_y], seed[node_x]
                    seed_prime_spread = self._fitness_function(set(seed))
                    
                    if seed_prime_spread > seed_spread:
                        r = 0
                        op1_success_num += 1
                        current_budget = (current_budget - 
                                        self.node_costs[pool[node_y]] + 
                                        self.node_costs[seed[node_x]])
                    else:
                        r += 1
                        op1_defeat_num += 1
                        seed[node_x], pool[node_y] = pool[node_y], seed[node_x]
                else:
                    # 操作2：低成本节点
                    # 找到满足条件的节点对
                    found = False
                    for _ in range(100):  # 最多尝试100次
                        node_x = random.randint(max(0, int(len(seed) / 2)), max(0, len(seed) - 1))
                        node_y = random.randint(0, len(pool) - 1)
                        
                        if (self.node_costs[pool[node_y]] - self.node_costs[seed[node_x]] < budget - current_budget
                            and pool[node_y] not in seed):
                            found = True
                            break
                    
                    if not found:
                        continue
                    
                    seed[node_x], pool[node_y] = pool[node_y], seed[node_x]
                    seed_prime_spread = self._fitness_function(set(seed))
                    
                    if seed_prime_spread > seed_spread:
                        r = 0
                        op2_success_num += 1
                        current_budget = (current_budget - 
                                        self.node_costs[pool[node_y]] + 
                                        self.node_costs[seed[node_x]])
                    else:
                        r += 1
                        op2_defeat_num += 1
                        seed[node_x], pool[node_y] = pool[node_y], seed[node_x]
            
            # 成功搜索10个节点后，进行成本检测
            if op1_success_num + op2_success_num - old_success_num > 10:
                current_budget = sum(self.node_costs[node] for node in seed)
                min_node_budget = self._low_bound_cost(pool)
                random.shuffle(pool)
                
                for node in pool:
                    if budget - current_budget < min_node_budget:
                        break
                    if node not in seed and self.node_costs[node] < budget - current_budget:
                        seed.append(node)
                        current_budget += self.node_costs[node]
                
                old_success_num = op1_success_num + op2_success_num
            
            # 更新温度
            t_t -= theta * math.log(r + 1)
            
            # 更新操作概率
            total_1 = op1_success_num + op1_defeat_num
            total_2 = op2_success_num + op2_defeat_num
            op_1 = (op1_success_num / total_1 + 0.0001) if total_1 > 0 else 0.0001
            op_2 = (op2_success_num / total_2 + 0.0001) if total_2 > 0 else 0.0001
            op = op_1 / (op_1 + op_2)
        
        # 最终成本检查
        current_budget = sum(self.node_costs[node] for node in seed)
        min_node_budget = self._low_bound_cost(pool)
        random.shuffle(pool)
        
        for node in pool:
            if budget - current_budget < min_node_budget:
                break
            if node not in seed and self.node_costs[node] < budget - current_budget:
                seed.append(node)
                current_budget += self.node_costs[node]
        
        return seed
    
    def _cost_greedy(self, node_dict: Dict[int, float], budget: float) -> Set[int]:
        """成本贪心选择
        
        cost_greedy
        """
        node_list = [(node, influence / self.node_costs[node]) 
                     for node, influence in node_dict.items()]
        node_list.sort(key=lambda x: x[1], reverse=True)
        
        seed = set()
        remaining_budget = budget
        
        for node, _ in node_list:
            if self.node_costs[node] < remaining_budget:
                seed.add(node)
                remaining_budget -= self.node_costs[node]
            
            if remaining_budget < 1:
                break
        
        return seed
    
    def run(self, k: int, **kwargs) -> Set[int]:
        """运行SADPEAAlgorithm算法
        
        两阶段算法执行流程：
            1. 进化算法阶段：双概率池的变异和交叉
            2. 模拟退火阶段：自适应温度调整优化
        
        这是预算约束的影响力最大化算法，内部自动计算预算。
        预算 = k * 平均节点成本
        
        Args:
            k: 种子节点数量
            
        Returns:
            选中的种子节点集合
            
        Example:
            >>> algo = SADPEAAlgorithm(graph, pop_size=20, max_iter=50)
            >>> seeds = algo.run(k=10)
            >>> print(f"选中 {len(seeds)} 个种子节点")
        """
        # 自动计算预算：k * 平均节点成本
        avg_cost = sum(self.node_costs.values()) / len(self.node_costs)
        budget = k * avg_cost
        
        # 阶段1：进化算法
        pool_2_set = self._evolution_stage(budget)
        
        # 阶段2：模拟退火
        # 使用影响力评估进行贪心初始化
        seed = self._cost_greedy(self.node_influence, budget)
        
        # 自适应模拟退火优化
        seed_list = self._adaptive_simulated_annealing(budget, seed, pool_2_set)
        
        # 保存种子节点集合并返回
        self.seeds = set(seed_list)
        return self.seeds
