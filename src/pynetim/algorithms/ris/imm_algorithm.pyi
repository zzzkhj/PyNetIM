"""IMM算法模块，提供Influence Maximization via Martingales算法"""

from typing import Set, Optional
from ..graph import IMGraph

class IMMAlgorithm:
    """Influence Maximization via Martingales 算法。

    IMM 是一个高效的 RIS 算法，使用鞅理论提供理论保证。
    相比 TIM，IMM 使用更少的 RR 集合达到相同的精度。

    该算法自动确定所需的 RR 集合数量，适合需要理论保证的场景。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        model: 扩散模型名称（'IC' 或 'LT'）。

    References:
        Youze Tang, Yanchen Shi, Xiaokui Xiao, 
        "Influence Maximization in Near-Linear Time: A Martingale Approach," 
        in Proc. ACM SIGMOD, 2015.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import IMMAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = IMMAlgorithm(graph, model='IC')
        >>> seeds = algo.run(k=10)
    """
    
    def __init__(self, graph: IMGraph, model: str, epsilon: float = 0.5,
                 l: int = 1, random_seed: Optional[int] = None,
                 verbose: bool = False) -> None:
        """初始化 IMM 算法。

        Args:
            graph: 输入图对象。
            model: 扩散模型名称，支持 'IC' 或 'LT'。
            epsilon: 近似误差参数，默认为 0.5。
            l: 偏差参数，默认为 1。
            random_seed: 随机种子，默认为 None（每次随机）。
            verbose: 是否显示关键过程日志，默认为 False。
        """
        ...
    
    def run(self, k: int) -> Set[int]:
        """执行 IMM 算法选择种子节点。

        IMM 自动确定所需的 RR 集合数量，以 (1-1/e-ε) 近似保证。

        Args:
            k: 需要选择的种子节点数量。

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
