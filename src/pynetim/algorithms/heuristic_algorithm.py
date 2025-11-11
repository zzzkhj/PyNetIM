import heapq
import time
from collections import defaultdict

from graph import IMGraph
from .base_algorithm import BaseAlgorithm
from diffusion_model import IndependentCascadeModel, LinearThresholdModel


class SingleDiscountAlgorithm(BaseAlgorithm):
    """
    简单度折扣启发式算法(Single Discount)用于选择影响力最大化种子节点。

    该算法通过逐步选择具有最高度数的节点作为种子，并对其邻居节点的度数进行折扣，
    以避免选择过多相互连接的节点。
    """

    def __init__(self, graph: IMGraph, diffusion_model=None):
        """
        初始化简单度折扣算法实例。

        Args:
            graph (IMGraph): 输入图对象
            diffusion_model: 扩散模型，此算法不使用该参数
        """
        super(SingleDiscountAlgorithm, self).__init__(graph, diffusion_model)
    
    def run(self, k: int):
        """
        运行简单度折扣算法选择k个种子节点。

        Args:
            k (int): 需要选择的种子节点数量

        Returns:
            list: 选择的种子节点列表
        """
        d = dict(self.graph.out_degree())  # 节点的初始度数
        seeds = []  # 存储种子节点
        selected = set()  # 记录已选择的种子节点

        # 使用最大堆存储 (-度数, 节点)，堆中的节点按度数排序
        heap = [(-d[v], v) for v in self.graph.nodes]
        heapq.heapify(heap)

        while len(seeds) < k:
            # 选择度数最大的节点作为种子节点
            _, u = heapq.heappop(heap)

            if u in selected:
                continue
            seeds.append(u)  # 将 u 添加到种子集合中
            selected.add(u)  # 标记 u 为已选择的节点

            # 对 u 的邻居节点的度数进行折扣
            for v in self.graph.out_neighbors(u):
                if v not in selected:
                    d[v] -= 1  # 将邻居节点 v 的度数减一
                    # 重新将折扣后的节点放回堆中（更新度数）
                    heapq.heappush(heap, (-d[v], v))

        self.seeds = seeds
        return seeds


class DegreeDiscountAlgorithm(BaseAlgorithm):
    """
    度折扣启发式算法(Degree Discount)用于选择影响力最大化种子节点。

    该算法是Single Discount的改进版本，考虑了邻居节点之间的影响关系，
    使用更复杂的折扣公式来更好地评估节点的边际影响力。
    """

    def __init__(self, graph: IMGraph, diffusion_model='IC'):
        """
        初始化度折扣算法实例。

        Args:
            graph (IMGraph): 输入图对象
            diffusion_model (str, optional): 扩散模型，默认为'IC'
        """
        super(DegreeDiscountAlgorithm, self).__init__(graph, diffusion_model)

    def run(self, k: int):
        """
        运行度折扣算法选择k个种子节点。

        Args:
            k (int): 需要选择的种子节点数量

        Returns:
            list: 选择的种子节点列表
        """
        d = dict(self.graph.out_degree())  # 节点的度
        dd = d.copy()  # 折扣度 degree discount
        t = defaultdict(int)  # t[v]: 节点v已被选为邻居种子的数量
        seeds = []  # 最终种子集合

        # 使用最大堆存储 (-折扣度, 节点)，注意 heapq 是最小堆，所以取负值
        heap = [(-dd[v], v) for v in self.graph.nodes]
        heapq.heapify(heap)
        selected = set()

        while len(seeds) < k:
            _, u = heapq.heappop(heap)
            if u in selected:
                continue
            seeds.append(u)
            selected.add(u)

            for v in self.graph.out_neighbors(u):
                if v in selected:
                    continue
                t[v] += 1
                # 获取边的权重
                weight = self.graph.edges[u, v]["weight"]
                # 更新折扣度
                dd[v] = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * weight
                heapq.heappush(heap, (-dd[v], v))  # 更新堆中的节点

        self.seeds = seeds
        return seeds
