import math
import random
import logging
from typing import List, Set, Dict, Callable, Tuple
from multiprocessing import Pool, cpu_count

from ..graph import IMGraph
from .base_algorithm import BaseAlgorithm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseRISAlgorithm(BaseAlgorithm):
    """
    最简版反向影响采样算法(Reverse Influence Sampling, RIS)

    说明:
    - 不自动估计采样数量
    - 用户指定RR集合数量(num_rr_sets)
    - 使用堆优化贪心选择(O(R log k)而非O(R k))
    - 支持IC/LT模型

    Attributes:
        graph (IMGraph): 输入图对象
        model (str): 扩散模型名称('IC'或'LT')
        nodes (list): 图中所有节点的列表
        rr_func (Callable): 用于生成RR集合的函数
        multi_process (bool): 是否启用多进程模式
        processes (int): 多进程模式下的进程数
        seed (int): 随机种子
    """

    def __init__(self, graph: IMGraph, diffusion_model: str = "IC", multi_process: bool = False, processes: int = None,
                 seed: int = None):
        """
        初始化基础RIS算法实例。

        Args:
            graph (IMGraph): 输入图对象
            diffusion_model (str): 扩散模型，支持'IC'或'LT'，默认为'IC'
            multi_process (bool, optional): 是否启用多进程模式，默认为False
            processes (int, optional): 多进程模式下的进程数，为None时使用CPU核心数
            seed (int, optional): 随机种子，默认为None
        """
        super(BaseRISAlgorithm, self).__init__(graph, diffusion_model)
        self.graph = graph
        self.model = diffusion_model.upper()
        self.nodes = list(graph.nodes)
        if self.model == "IC":
            self.rr_func = self._sample_rr_ic
        elif self.model == "LT":
            self.rr_func = self._sample_rr_lt
        else:
            raise ValueError("模型必须是'IC'或'LT'")

        if multi_process:
            if processes is None:
                processes = cpu_count()
            self.processes = processes
        self.multi_process = multi_process

        self.seed = seed
        random.seed(self.seed)

    def _sample_rr_ic(self, start_node: int) -> Set[int]:
        """
        IC模型：从起始节点反向BFS采样可达节点。

        Args:
            start_node (int): 起始节点

        Returns:
            Set[int]: 反向可达节点集合
        """
        active = {start_node}
        queue = [start_node]

        while queue:
            u = queue.pop(0)
            for v in self.graph.in_neighbors(u):
                if v not in active and random.random() <= self.graph.edges[v, u]['weight']:
                    active.add(v)
                    queue.append(v)
        return active

    def _sample_rr_lt(self, start_node: int) -> Set[int]:
        """
        LT模型：从起始节点反向随机走一条路径。

        Args:
            start_node (int): 起始节点

        Returns:
            Set[int]: 反向可达节点集合
        """
        active = {start_node}
        current = start_node
        nodes = list(self.graph.nodes)

        while current in nodes:
            in_neighbors = list(self.graph.in_neighbors(current))
            if not in_neighbors:
                break
            # 随机选择一个前驱（等概率）
            pred = random.choice(in_neighbors)
            if pred not in active:
                active.add(pred)
                current = pred
            else:
                break  # 遇到已激活节点，停止
        return active

    def _generate_rr_sets_single_process(self, num_rr_sets: int) -> List[Set[int]]:
        """
        在单个进程中生成RR集合。

        Args:
            num_rr_sets (int): 需要生成的RR集合数量

        Returns:
            List[Set[int]]: RR集合列表
        """
        rr_sets = []
        nodes = self.nodes
        for i in range(num_rr_sets):
            random.seed(self.seed)
            v = random.choice(nodes)
            rr_set = self.rr_func(v)
            rr_sets.append(rr_set)
        return rr_sets

    def _generate_rr_sets(self, num_rr_sets: int) -> List[Set[int]]:
        """
        生成指定数量的RR集合，支持单进程和多进程模式。

        Args:
            num_rr_sets (int): 需要生成的RR集合数量

        Returns:
            List[Set[int]]: RR集合列表
        """
        if self.multi_process:
            # 每个进程生成 num_rr_sets / processes 个RR集合
            rr_sets_per_worker = int(num_rr_sets / self.processes)
            remaining_rr_sets = num_rr_sets - rr_sets_per_worker * self.processes

            with Pool(processes=self.processes) as pool:
                # 为每个进程准备参数
                args = [(rr_sets_per_worker, ) for _ in range(self.processes)]
                # 如果有余数，分配给第一个进程处理
                if remaining_rr_sets > 0 and args:
                    args[0] = (rr_sets_per_worker + remaining_rr_sets)

                results = pool.starmap(self._generate_rr_sets_single_process, args)

            # 合并所有进程的结果
            rr_sets = []
            for result in results:
                rr_sets.extend(result)

            logger.info(f"已生成 {len(rr_sets)} 个RR集合（多进程模式）")
        else:
            # 单进程模式（原始实现）
            rr_sets = []
            for _ in range(num_rr_sets):
                v = random.choice(self.nodes)
                rr_set = self.rr_func(v)
                rr_sets.append(rr_set)
            logger.info(f"已生成 {len(rr_sets)} 个RR集合（单进程模式）")

        return rr_sets

    def _node_selection(self, rr_sets: List[Set[int]], k: int) -> Tuple[Set[int], float]:
        """
        节点选择算法，返回选出的种子集合和覆盖比例。

        Args:
            rr_sets (List[Set[int]]): RR集合列表
            k (int): 需要选择的种子节点数量

        Returns:
            Tuple[Set[int], float]: 选出的种子节点集合和覆盖比例
        """
        rr_degree = {node: 0 for node in self.graph.nodes}
        node_to_rr_idx: Dict[int, List[int]] = {}

        # 统计每个节点出现在多少个RR集合中
        for rr_idx, rr_set in enumerate(rr_sets):
            for node in rr_set:
                rr_degree[node] += 1
                node_to_rr_idx.setdefault(node, []).append(rr_idx)

        Sk = set()
        matched_count = 0

        for _ in range(k):
            # 找当前覆盖最多未覆盖RR集合的节点
            max_node = max(self.graph.nodes, key=lambda x: rr_degree[x])
            if rr_degree[max_node] == 0:
                break

            Sk.add(max_node)
            matched_count += len(node_to_rr_idx[max_node])

            # 删除已被该节点覆盖的RR集合
            for rr_idx in node_to_rr_idx[max_node]:
                for node_in_rr in rr_sets[rr_idx]:
                    rr_degree[node_in_rr] -= 1
                    node_to_rr_idx[node_in_rr].remove(rr_idx)

        coverage_ratio = matched_count / len(rr_sets) if rr_sets else 0.0
        return Sk, coverage_ratio

    def run(self, k: int, num_rr_sets: int) -> List[int]:
        """
        执行简单RIS算法。

        Args:
            k (int): 种子集合大小
            num_rr_sets (int): RR集合采样数量（越大越准）

        Returns:
            Set[int]: 选出的k个种子节点
        """
        if k <= 0 or num_rr_sets <= 0:
            return []

        logger.info(f"SimpleRIS: k={k}, num_rr_sets={num_rr_sets}, model={self.model}")

        # Step 1: 生成RR集合
        rr_sets = self._generate_rr_sets(num_rr_sets)

        # 选择节点
        seeds, _ = self._node_selection(rr_sets, k)

        return list(seeds)


class IMMAlgorithm(BaseRISAlgorithm):
    """
    基于鞅的影响力最大化算法(Influence Maximization via Martingales, IMM)

    支持IC和LT两种扩散模型

    Attributes:
        eps (float): 近似参数ε
        l (float): 失败概率参数l
    """

    def __init__(self,
                 graph: IMGraph,
                 diffusion_model: str = "IC",
                 eps: float = 0.5,
                 l: int = 1,
                 multi_process: bool = False, processes: int = None,
                 seed: int = None):
        """
        初始化IMM算法实例。

        Args:
            graph (IMGraph): 有向带权图
            diffusion_model (str): 扩散模型，支持'IC'或'LT'，默认为'IC'
            eps (float): 近似参数ε，默认为0.5
            l (int): 失败概率参数l，默认为1
            multi_process (bool, optional): 是否启用多进程模式，默认为False
            processes (int, optional): 多进程模式下的进程数，为None时使用CPU核心数
            seed (int, optional): 随机种子，默认为None
        """
        super(IMMAlgorithm, self).__init__(graph, diffusion_model, multi_process, processes, seed)
        self.eps = eps
        self.l = l * (1 + math.log(2) / math.log(graph.number_of_nodes))

    @staticmethod
    def _log_binomial(n: int, k: int) -> float:
        """
        计算组合数的对数log(C(n,k))。

        Args:
            n (int): 总数
            k (int): 选择数

        Returns:
            float: 组合数的对数
        """
        if k == 0:
            return 0.0
        res = 0.0
        for i in range(n - k + 1, n + 1):
            res += math.log(i)
        for i in range(1, k + 1):
            res -= math.log(i)
        return res

    def _sampling(self, k: int) -> List[Set[int]]:
        """
        IMM采样主过程。

        Args:
            k (int): 需要选择的种子节点数量

        Returns:
            List[Set[int]]: 采样得到的RR集合列表
        """
        n = len(self.nodes)
        R: List[Set[int]] = []
        LB = 1.0

        eps_p = self.eps * math.sqrt(2)
        log_binom = self._log_binomial(n, k)
        lambda_p = (2 + 2 * eps_p / 3) * (
                log_binom + self.l * math.log(n) + math.log(math.log2(n))
        ) * n / (eps_p ** 2)

        # Phase 1: 二分查找下界LB
        for i in range(1, int(math.log2(n)) + 1):
            x = n / (2 ** i)
            theta_i = int(lambda_p / x)

            while len(R) < theta_i:
                v = random.choice(list(self.graph.nodes))
                R.append(self.rr_func(v))

            _, F = self._node_selection(R, k)
            if n * F >= (1 + eps_p) * x:
                LB = n * F / (1 + eps_p)
                break

        # Phase 2: 计算最终需要的采样数量theta*
        alpha = math.sqrt(self.l * math.log(n) + math.log(2))
        beta = math.sqrt((1 - 1 / math.e) * (log_binom + self.l * math.log(n) + math.log(2)))
        lambda_star = 2 * n * ((1 - 1 / math.e) * alpha + beta) ** 2 / (self.eps ** 2)
        theta_star = lambda_star / LB

        while len(R) < theta_star:
            v = random.choice(list(self.graph.nodes))
            R.append(self.rr_func(v))

        return R

    def run(self, k: int) -> List[int]:
        """
        执行IMM算法，返回大小为k的种子集合。

        Args:
            k (int): 需要选择的种子节点数量

        Returns:
            Set[int]: 选出的种子节点集合
        """
        logger.info(f"开始采样阶段...")
        R = self._sampling(k)
        logger.info(f"反向可达集数量 |R| = {len(R)}")

        logger.info(f"开始节点选择阶段...")
        seeds, _ = self._node_selection(R, k)
        # logger.info(f"选出的种子节点: {sorted(Sk)}")
        return list(seeds)
