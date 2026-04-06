import heapq
from typing import List, Set, TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from ..graph import IMGraph
    from ..diffusion_model import IndependentCascadeModel, LinearThresholdModel

from .base_algorithm import BaseAlgorithm


class GreedyAlgorithm(BaseAlgorithm):
    """贪婪算法。

    通过迭代地选择具有最大边际影响力的节点作为种子节点，
    直到选够 k 个种子节点。每次选择都需要计算所有候选节点的边际增益。

    该算法精度高，但速度较慢，适合小规模图或需要高精度的场景。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        diffusion_model: 扩散模型类。

    References:
        Kempe, D., Kleinberg, J., & Tardos, É. (2003). Maximizing the spread 
        of influence through a social network. KDD, 137-146.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import GreedyAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = GreedyAlgorithm(graph, diffusion_model='IC')
        >>> seeds = algo.run(k=10, rounds=1000, num_threads=4)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = 'IC'):
        """初始化贪婪算法。

        Args:
            graph: 输入图对象。
            diffusion_model: 扩散模型名称，支持 'IC' 或 'LT'，默认为 'IC'。
        """
        super().__init__(graph, diffusion_model)

    def run(self, k: int, rounds: int = 1000, num_threads: int = 4, 
            show_progress: bool = True, seed: int = None) -> Set[int]:
        """运行贪婪算法选择种子节点。

        Args:
            k: 需要选择的种子节点数量。
            rounds: 每次计算边际增益的蒙特卡洛模拟次数，默认 1000。
            num_threads: 多线程数，默认 4。
            show_progress: 是否显示进度条，默认 True。
            seed: 模拟的随机种子，默认 None。

        Returns:
            Set[int]: 选中的种子节点集合。

        Note:
            时间复杂度为 O(k * n * rounds)，其中 n 为节点数。
            对于大规模图，建议使用 CELFAlgorithm 或 IMMAlgorithm。
        """
        seeds: Set[int] = set()
        nodes = set(range(self.graph.num_nodes))

        outer_pbar = tqdm(range(k), desc="选择种子节点", disable=not show_progress)

        for i in outer_pbar:
            best_node = None
            best_gain = -1.0

            node_list = list(nodes - seeds)
            inner_pbar = tqdm(node_list, desc=f"评估候选节点({i+1}/{k})",
                             leave=False, disable=not show_progress)

            for node in inner_pbar:
                model_with = self.diffusion_model(self.graph, seeds | {node})
                avg_with = model_with.run_monte_carlo_diffusion(rounds, num_threads)
                
                model_without = self.diffusion_model(self.graph, seeds)
                avg_without = model_without.run_monte_carlo_diffusion(rounds, num_threads)
                
                marginal_gain = avg_with - avg_without

                if marginal_gain > best_gain:
                    best_gain = marginal_gain
                    best_node = node

                if show_progress:
                    inner_pbar.set_postfix({'当前最佳增益': f'{best_gain:.4f}'})

            seeds.add(best_node)
            if show_progress:
                outer_pbar.set_description(f"已选中节点: {best_node}")

        self.seeds = seeds
        return seeds


class CELFAlgorithm(BaseAlgorithm):
    """CELF 算法。

    CELF (Cost-Effective Lazy Forward) 是贪婪算法的优化版本，
    利用边际增益的子模特性减少计算量，通过优先队列维护节点的边际增益，
    避免重复计算已失效的边际增益。

    该算法精度高，比贪婪算法快 2-7 倍，适合中等规模图。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        diffusion_model: 扩散模型类。

    References:
        Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J., 
        & Glance, N. (2007). Cost-effective outbreak detection in networks. 
        KDD, 420-429.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import CELFAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = CELFAlgorithm(graph, diffusion_model='IC')
        >>> seeds = algo.run(k=10, rounds=1000, num_threads=4)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = 'IC'):
        """初始化 CELF 算法。

        Args:
            graph: 输入图对象。
            diffusion_model: 扩散模型名称，支持 'IC' 或 'LT'，默认为 'IC'。
        """
        super().__init__(graph, diffusion_model)

    def run(self, k: int, rounds: int = 1000, num_threads: int = 4,
            show_progress: bool = True, seed: int = None) -> Set[int]:
        """运行 CELF 算法选择种子节点。

        Args:
            k: 需要选择的种子节点数量。
            rounds: 蒙特卡洛模拟次数，默认 1000。
            num_threads: 多线程数，默认 4。
            show_progress: 是否显示进度条，默认 True。
            seed: 模拟的随机数种子，默认 None。

        Returns:
            Set[int]: 选中的种子节点集合。

        Note:
            利用子模特性，CELF 比贪婪算法快 2-7 倍，同时保证相同的近似比。
        """
        seeds: Set[int] = set()
        nodes = set(range(self.graph.num_nodes))
        heap = []

        node_list = list(nodes)
        init_pbar = tqdm(node_list, desc="初始化边际增益", disable=not show_progress)

        for node in init_pbar:
            model = self.diffusion_model(self.graph, {node})
            mg = model.run_monte_carlo_diffusion(rounds, num_threads)
            heap.append((-mg, node, 0))

        heapq.heapify(heap)

        main_pbar = tqdm(range(k), desc="选择种子节点", disable=not show_progress)

        for _ in main_pbar:
            while True:
                neg_gain, node, last_s_size = heapq.heappop(heap)

                if last_s_size == len(seeds):
                    seeds.add(node)
                    if show_progress:
                        main_pbar.set_description(f"已选中节点: {node}")
                    break
                else:
                    model_with = self.diffusion_model(self.graph, seeds | {node})
                    avg_with = model_with.run_monte_carlo_diffusion(rounds, num_threads)
                    
                    model_without = self.diffusion_model(self.graph, seeds)
                    avg_without = model_without.run_monte_carlo_diffusion(rounds, num_threads)
                    
                    marginal_gain = avg_with - avg_without
                    heapq.heappush(heap, (-marginal_gain, node, len(seeds)))

        self.seeds = seeds
        return seeds
