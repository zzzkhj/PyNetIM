"""RIS算法基类模块，提供Reverse Influence Sampling的基础框架"""

from typing import Set, Optional
from ..graph import IMGraph

class BaseRISAlgorithm:
    """RIS (Reverse Influence Sampling) 算法基类。

    RIS 是一类影响力最大化算法的基础框架，通过反向采样来估计节点的影响力。
    主要步骤：
    1. 生成反向可达集 (RR Sets)
    2. 使用贪心策略选择覆盖最多 RR 集合的节点

    该类为 TIM、IMM、OPIM 等算法提供统一接口。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        model: 扩散模型名称（'IC' 或 'LT'）。

    References:
        Chong Wang, Wei Chen, Yajun Wang, 
        "Scalable Influence Maximization for Independent Cascade Model in Large Social Networks," 
        Data Mining and Knowledge Discovery, 2012.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import BaseRISAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = BaseRISAlgorithm(graph, model='IC')
        >>> seeds = algo.run(k=10, num_rr_sets=10000)
    """
    
    def __init__(self, graph: IMGraph, model: str, random_seed: Optional[int] = None,
                 verbose: bool = False) -> None:
        """初始化 RIS 算法基类。

        Args:
            graph: 输入图对象。
            model: 扩散模型名称，支持 'IC' 或 'LT'。
            random_seed: 随机种子，默认为 None（每次随机）。
            verbose: 是否显示关键过程日志，默认为 False。
        """
        ...
    
    def run(self, k: int, num_rr_sets: int) -> Set[int]:
        """执行 RIS 算法选择种子节点。

        使用固定数量的 RR 集合进行贪心选择。

        Args:
            k: 需要选择的种子节点数量。
            num_rr_sets: RR 集合采样数量。

        Returns:
            Set[int]: 选中的种子节点集合。
        """
        ...
    
    def get_seeds(self) -> Set[int]:
        """获取最后一次运行选出的种子集合。

        Returns:
            Set[int]: 种子节点集合。
        """
        ...
