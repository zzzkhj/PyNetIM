import math
import random
import logging
from typing import List, Set, Dict, Tuple, TYPE_CHECKING
from multiprocessing import Pool, cpu_count

if TYPE_CHECKING:
    from ..graph import IMGraph

from .base_algorithm import BaseAlgorithm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseRISAlgorithm(BaseAlgorithm):
    """反向影响采样算法基类。

    基于反向影响采样（Reverse Influence Sampling, RIS）的影响力最大化算法。
    用户指定 RR 集合数量，使用堆优化贪心选择。

    该算法速度快且可扩展性好，适合大规模图。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        model: 扩散模型名称（'IC' 或 'LT'）。
        nodes: 图中所有节点的列表。
        rr_func: 用于生成 RR 集合的函数。
        multi_process: 是否启用多进程模式。
        processes: 多进程模式下的进程数。

    References:
        Borgs, C., Brautbar, M., Chitnis, N., & Tardos, É. (2014). 
        Maximizing social influence in nearly optimal time. SODA, 946-957.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import BaseRISAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = BaseRISAlgorithm(graph, diffusion_model='IC')
        >>> seeds = algo.run(k=10, num_rr_sets=10000)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = "IC", 
                 multi_process: bool = False, processes: int = None, seed: int = None):
        """初始化 RIS 算法。

        Args:
            graph: 输入图对象。
            diffusion_model: 扩散模型名称，支持 'IC' 或 'LT'，默认为 'IC'。
            multi_process: 是否启用多进程模式，默认为 False。
            processes: 多进程模式下的进程数，为 None 时使用 CPU 核心数。
            seed: 随机种子，默认为 None。

        Raises:
            ValueError: 当 diffusion_model 不是 'IC' 或 'LT' 时抛出。
        """
        super().__init__(graph, diffusion_model)
        self.model = diffusion_model.upper()
        self.nodes = list(range(graph.num_nodes))
        
        if self.model == "IC":
            self.rr_func = self._sample_rr_ic
        elif self.model == "LT":
            self.rr_func = self._sample_rr_lt
        else:
            raise ValueError("模型必须是 'IC' 或 'LT'")

        self.multi_process = multi_process
        if multi_process:
            self.processes = processes if processes else cpu_count()
        else:
            self.processes = 1

        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def _sample_rr_ic(self, start_node: int) -> Set[int]:
        """IC 模型下从起始节点反向 BFS 采样可达节点。

        Args:
            start_node: 起始节点。

        Returns:
            Set[int]: 反向可达节点集合。
        """
        active = {start_node}
        queue = [start_node]

        while queue:
            u = queue.pop(0)
            in_neighbors = self.graph.in_neighbors(u)
            for v in in_neighbors:
                if v not in active:
                    weight = self.graph.get_edge_weight(v, u)
                    if random.random() <= weight:
                        active.add(v)
                        queue.append(v)
        return active

    def _sample_rr_lt(self, start_node: int) -> Set[int]:
        """LT 模型下从起始节点反向随机走一条路径。

        Args:
            start_node: 起始节点。

        Returns:
            Set[int]: 反向可达节点集合。
        """
        active = {start_node}
        current = start_node

        while True:
            in_neighbors = self.graph.in_neighbors(current)
            if not in_neighbors:
                break
            pred = random.choice(in_neighbors)
            if pred not in active:
                active.add(pred)
                current = pred
            else:
                break
        return active

    def _generate_rr_sets_single_process(self, num_rr_sets: int) -> List[Set[int]]:
        """在单个进程中生成 RR 集合。

        Args:
            num_rr_sets: 需要生成的 RR 集合数量。

        Returns:
            List[Set[int]]: RR 集合列表。
        """
        rr_sets = []
        for _ in range(num_rr_sets):
            v = random.choice(self.nodes)
            rr_set = self.rr_func(v)
            rr_sets.append(rr_set)
        return rr_sets

    def _generate_rr_sets(self, num_rr_sets: int) -> List[Set[int]]:
        """生成指定数量的 RR 集合，支持单进程和多进程模式。

        Args:
            num_rr_sets: 需要生成的 RR 集合数量。

        Returns:
            List[Set[int]]: RR 集合列表。
        """
        if self.multi_process:
            rr_sets_per_worker = int(num_rr_sets / self.processes)
            remaining_rr_sets = num_rr_sets - rr_sets_per_worker * self.processes

            with Pool(processes=self.processes) as pool:
                args = [(rr_sets_per_worker,) for _ in range(self.processes)]
                if remaining_rr_sets > 0 and args:
                    args[0] = (rr_sets_per_worker + remaining_rr_sets,)

                results = pool.starmap(self._generate_rr_sets_single_process, args)

            rr_sets = []
            for result in results:
                rr_sets.extend(result)

            logger.info(f"已生成 {len(rr_sets)} 个 RR 集合（多进程模式）")
        else:
            rr_sets = []
            for _ in range(num_rr_sets):
                v = random.choice(self.nodes)
                rr_set = self.rr_func(v)
                rr_sets.append(rr_set)
            logger.info(f"已生成 {len(rr_sets)} 个 RR 集合（单进程模式）")

        return rr_sets

    def _node_selection(self, rr_sets: List[Set[int]], k: int) -> Tuple[Set[int], float]:
        """节点选择算法，返回选出的种子集合和覆盖比例。

        Args:
            rr_sets: RR 集合列表。
            k: 需要选择的种子节点数量。

        Returns:
            Tuple[Set[int], float]: 包含两个元素：
                - 选出的种子节点集合
                - 覆盖比例
        """
        rr_degree = {node: 0 for node in self.nodes}
        node_to_rr_idx: Dict[int, List[int]] = {}

        for rr_idx, rr_set in enumerate(rr_sets):
            for node in rr_set:
                rr_degree[node] += 1
                node_to_rr_idx.setdefault(node, []).append(rr_idx)

        Sk: Set[int] = set()
        matched_count = 0

        for _ in range(k):
            max_node = max(self.nodes, key=lambda x: rr_degree[x])
            if rr_degree[max_node] == 0:
                break

            Sk.add(max_node)
            matched_count += len(node_to_rr_idx[max_node])

            for rr_idx in node_to_rr_idx[max_node]:
                for node_in_rr in rr_sets[rr_idx]:
                    rr_degree[node_in_rr] -= 1
                    node_to_rr_idx[node_in_rr].remove(rr_idx)

        coverage_ratio = matched_count / len(rr_sets) if rr_sets else 0.0
        return Sk, coverage_ratio

    def run(self, k: int, num_rr_sets: int) -> Set[int]:
        """执行 RIS 算法选择种子节点。

        Args:
            k: 需要选择的种子节点数量。
            num_rr_sets: RR 集合采样数量，越大越准确。

        Returns:
            Set[int]: 选中的种子节点集合。

        Note:
            RR 集合数量建议设置为图节点数的 100-1000 倍。
        """
        if k <= 0 or num_rr_sets <= 0:
            return set()

        logger.info(f"SimpleRIS: k={k}, num_rr_sets={num_rr_sets}, model={self.model}")

        rr_sets = self._generate_rr_sets(num_rr_sets)
        seeds, _ = self._node_selection(rr_sets, k)

        self.seeds = seeds
        return seeds


class IMMAlgorithm(BaseRISAlgorithm):
    """基于鞅的影响力最大化算法。

    IMM (Influence Maximization via Martingales) 是 RIS 算法的改进版本，
    自动估计最优采样数量，在保证理论精度的同时提高效率。

    该算法速度快、精度高、可扩展性好，适合大规模图。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        model: 扩散模型名称（'IC' 或 'LT'）。
        eps: 近似参数 ε。
        l: 失败概率参数。

    References:
        Tang, Y., Xiao, X., & Shi, Y. (2015). Influence maximization: 
        Near-optimal time complexity meets practical efficiency. SIGMOD, 75-86.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import IMMAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = IMMAlgorithm(graph, diffusion_model='IC', eps=0.5)
        >>> seeds = algo.run(k=10)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = "IC",
                 eps: float = 0.5, l: int = 1,
                 multi_process: bool = False, processes: int = None, seed: int = None):
        """初始化 IMM 算法。

        Args:
            graph: 输入图对象。
            diffusion_model: 扩散模型名称，支持 'IC' 或 'LT'，默认为 'IC'。
            eps: 近似参数 ε，默认为 0.5。
            l: 失败概率参数，默认为 1。
            multi_process: 是否启用多进程模式，默认为 False。
            processes: 多进程模式下的进程数，为 None 时使用 CPU 核心数。
            seed: 随机种子，默认为 None。
        """
        super().__init__(graph, diffusion_model, multi_process, processes, seed)
        self.eps = eps
        self.l = l * (1 + math.log(2) / math.log(graph.num_nodes))

    @staticmethod
    def _log_binomial(n: int, k: int) -> float:
        """计算组合数的对数 log(C(n,k))。

        Args:
            n: 总数。
            k: 选择数。

        Returns:
            float: 组合数的对数。
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
        """IMM 采样主过程。

        Args:
            k: 需要选择的种子节点数量。

        Returns:
            List[Set[int]]: 采样得到的 RR 集合列表。
        """
        n = len(self.nodes)
        R: List[Set[int]] = []
        LB = 1.0

        eps_p = self.eps * math.sqrt(2)
        log_binom = self._log_binomial(n, k)
        lambda_p = (2 + 2 * eps_p / 3) * (
                log_binom + self.l * math.log(n) + math.log(math.log2(n))
        ) * n / (eps_p ** 2)

        for i in range(1, int(math.log2(n)) + 1):
            x = n / (2 ** i)
            theta_i = int(lambda_p / x)

            while len(R) < theta_i:
                v = random.choice(self.nodes)
                R.append(self.rr_func(v))

            _, F = self._node_selection(R, k)
            if n * F >= (1 + eps_p) * x:
                LB = n * F / (1 + eps_p)
                break

        alpha = math.sqrt(self.l * math.log(n) + math.log(2))
        beta = math.sqrt((1 - 1 / math.e) * (log_binom + self.l * math.log(n) + math.log(2)))
        lambda_star = 2 * n * ((1 - 1 / math.e) * alpha + beta) ** 2 / (self.eps ** 2)
        theta_star = lambda_star / LB

        while len(R) < theta_star:
            v = random.choice(self.nodes)
            R.append(self.rr_func(v))

        return R

    def run(self, k: int) -> Set[int]:
        """执行 IMM 算法选择种子节点。

        Args:
            k: 需要选择的种子节点数量。

        Returns:
            Set[int]: 选中的种子节点集合。

        Note:
            IMM 会自动估计最优采样数量，无需手动指定。
        """
        logger.info(f"开始采样阶段...")
        R = self._sampling(k)
        logger.info(f"反向可达集数量 |R| = {len(R)}")

        logger.info(f"开始节点选择阶段...")
        seeds, _ = self._node_selection(R, k)
        
        self.seeds = seeds
        return seeds
