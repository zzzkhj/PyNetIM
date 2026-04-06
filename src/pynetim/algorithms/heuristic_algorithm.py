import heapq
from typing import List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import IMGraph

from .base_algorithm import BaseAlgorithm


class SingleDiscountAlgorithm(BaseAlgorithm):
    """简单度折扣启发式算法。

    通过逐步选择具有最高度数的节点作为种子，并对其邻居节点的度数进行折扣，
    以避免选择过多相互连接的节点。

    该算法速度快，适合大规模图的快速种子选择。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。

    References:
        Chen, W., Wang, Y., & Yang, S. (2009). Efficient influence maximization 
        in social networks. KDD, 199-208.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import SingleDiscountAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = SingleDiscountAlgorithm(graph)
        >>> seeds = algo.run(k=10)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = None):
        """初始化简单度折扣算法。

        Args:
            graph: 输入图对象。
            diffusion_model: 扩散模型（此算法不使用该参数）。
        """
        super().__init__(graph, diffusion_model)
    
    def run(self, k: int) -> Set[int]:
        """运行算法选择种子节点。

        Args:
            k: 需要选择的种子节点数量。

        Returns:
            Set[int]: 选中的种子节点集合。
        """
        d = {v: self.graph.out_degree(v) for v in range(self.graph.num_nodes)}
        seeds: Set[int] = set()
        selected: Set[int] = set()

        heap = [(-d[v], v) for v in range(self.graph.num_nodes)]
        heapq.heapify(heap)

        while len(seeds) < k:
            _, u = heapq.heappop(heap)

            if u in selected:
                continue
            seeds.add(u)
            selected.add(u)

            for v, _ in self.graph.out_neighbors_with_weights(u):
                if v not in selected:
                    d[v] -= 1
                    heapq.heappush(heap, (-d[v], v))

        self.seeds = seeds
        return seeds


class DegreeDiscountAlgorithm(BaseAlgorithm):
    """度折扣启发式算法。

    是 SingleDiscountAlgorithm 的改进版本，考虑了邻居节点之间的影响关系，
    使用更复杂的折扣公式来更好地评估节点的边际影响力。

    该算法速度快且效果较好，适合大规模图的种子选择。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。

    References:
        Chen, W., Wang, Y., & Yang, S. (2009). Efficient influence maximization 
        in social networks. KDD, 199-208.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import DegreeDiscountAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = DegreeDiscountAlgorithm(graph, diffusion_model='IC')
        >>> seeds = algo.run(k=10)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = 'IC'):
        """初始化度折扣算法。

        Args:
            graph: 输入图对象。
            diffusion_model: 扩散模型名称，支持 'IC' 或 'LT'，默认为 'IC'。
        """
        super().__init__(graph, diffusion_model)

    def run(self, k: int) -> Set[int]:
        """运行算法选择种子节点。

        Args:
            k: 需要选择的种子节点数量。

        Returns:
            Set[int]: 选中的种子节点集合。
        """
        from collections import defaultdict
        
        d = {v: self.graph.out_degree(v) for v in range(self.graph.num_nodes)}
        dd = d.copy()
        t = defaultdict(int)
        seeds: Set[int] = set()

        heap = [(-dd[v], v) for v in range(self.graph.num_nodes)]
        heapq.heapify(heap)
        selected: Set[int] = set()

        while len(seeds) < k:
            _, u = heapq.heappop(heap)
            if u in selected:
                continue
            seeds.add(u)
            selected.add(u)

            for v, weight in self.graph.out_neighbors_with_weights(u):
                if v in selected:
                    continue
                t[v] += 1
                dd[v] = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * weight
                heapq.heappush(heap, (-dd[v], v))

        self.seeds = seeds
        return seeds
