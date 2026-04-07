import heapq
from typing import List, Set, TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from ...graph import IMGraph
    from ...diffusion_model import IndependentCascadeModel, LinearThresholdModel

from ..base_algorithm import BaseAlgorithm


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


class CELFPlusAlgorithm(BaseAlgorithm):
    """CELF++ 算法。

    CELF++ 是 CELF 算法的改进版本，进一步利用子模特性减少计算量。
    在更新节点边际增益时，同时考虑当前最佳候选节点，如果该节点增益仍然最大，
    则直接选择它，避免额外的重新计算。

    该算法精度高，比 CELF 更快，适合中等规模图。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        diffusion_model: 扩散模型类。

    References:
        Goyal, A., Lu, W., & Lakshmanan, L. V. (2011). 
        CELF++: Optimizing the greedy algorithm for influence maximization 
        in social networks. WWW, 47-48.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import CELFPlusAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = CELFPlusAlgorithm(graph, diffusion_model='IC')
        >>> seeds = algo.run(k=10, rounds=1000, num_threads=4)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = 'IC'):
        """初始化 CELF++ 算法。

        Args:
            graph: 输入图对象。
            diffusion_model: 扩散模型名称，支持 'IC' 或 'LT'，默认为 'IC'。
        """
        super().__init__(graph, diffusion_model)

    def run(self, k: int, rounds: int = 1000, num_threads: int = 4,
            show_progress: bool = True, seed: int = None) -> Set[int]:
        """运行 CELF++ 算法选择种子节点。

        Args:
            k: 需要选择的种子节点数量。
            rounds: 蒙特卡洛模拟次数，默认 1000。
            num_threads: 多线程数，默认 4。
            show_progress: 是否显示进度条，默认 True。
            seed: 模拟的随机数种子，默认 None。

        Returns:
            Set[int]: 选中的种子节点集合。

        Note:
            CELF++ 比 CELF 更快，同时保证相同的近似比。
        """
        seeds: Set[int] = set()
        nodes = set(range(self.graph.num_nodes))
        
        mg1 = {}
        prev_best = None
        mg2 = {}
        flag = {}
        
        node_list = list(nodes)
        init_pbar = tqdm(node_list, desc="初始化边际增益", disable=not show_progress)
        
        for node in init_pbar:
            model = self.diffusion_model(self.graph, {node})
            mg1[node] = model.run_monte_carlo_diffusion(rounds, num_threads)
            flag[node] = 0
        
        prev_best = max(mg1, key=mg1.get)
        
        for node in node_list:
            if node != prev_best:
                model = self.diffusion_model(self.graph, {prev_best, node})
                mg2[node] = model.run_monte_carlo_diffusion(rounds, num_threads)
            else:
                mg2[node] = mg1[node]
        
        heap = []
        for node in node_list:
            heapq.heappush(heap, (-mg1[node], node, 0, mg2[node]))
        
        main_pbar = tqdm(range(k), desc="选择种子节点", disable=not show_progress)
        
        for _ in main_pbar:
            while True:
                neg_gain, node, last_s_size, mg2_val = heapq.heappop(heap)
                gain = -neg_gain
                
                if last_s_size == len(seeds):
                    seeds.add(node)
                    if show_progress:
                        main_pbar.set_description(f"已选中节点: {node}")
                    
                    prev_best = None
                    max_mg1 = -1
                    for n, g in mg1.items():
                        if n not in seeds and g > max_mg1:
                            max_mg1 = g
                            prev_best = n
                    break
                else:
                    if flag[node] == len(seeds):
                        new_mg1 = mg2_val
                    else:
                        model_with = self.diffusion_model(self.graph, seeds | {node})
                        avg_with = model_with.run_monte_carlo_diffusion(rounds, num_threads)
                        
                        model_without = self.diffusion_model(self.graph, seeds)
                        avg_without = model_without.run_monte_carlo_diffusion(rounds, num_threads)
                        
                        new_mg1 = avg_with - avg_without
                    
                    if prev_best is not None and prev_best != node:
                        model_with = self.diffusion_model(self.graph, seeds | {prev_best, node})
                        avg_with = model_with.run_monte_carlo_diffusion(rounds, num_threads)
                        
                        model_without = self.diffusion_model(self.graph, seeds | {prev_best})
                        avg_without = model_without.run_monte_carlo_diffusion(rounds, num_threads)
                        
                        new_mg2 = avg_with - avg_without
                    else:
                        new_mg2 = new_mg1
                    
                    mg1[node] = new_mg1
                    mg2[node] = new_mg2
                    flag[node] = len(seeds)
                    
                    heapq.heappush(heap, (-new_mg1, node, len(seeds), new_mg2))
        
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
