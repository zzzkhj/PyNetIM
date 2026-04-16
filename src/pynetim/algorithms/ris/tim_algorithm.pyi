"""TIM算法模块，提供Two-phase Influence Maximization算法"""

from typing import Set, Optional
from ..graph import IMGraph

class TIMAlgorithm:
    """Two-phase Influence Maximization 算法。

    TIM 是一个经典的 RIS 算法，分两个阶段：
    1. 估计最优解的下界
    2. 生成足够数量的 RR 集合并贪心选择

    该算法提供 (1-1/e-ε) 近似保证。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        model: 扩散模型名称（'IC' 或 'LT'）。

    References:
        Youze Tang, Xiaokui Xiao, Yanchen Shi, 
        "Influence Maximization: Near-Optimal Time Complexity Meets Practical Efficiency," 
        in Proc. ACM SIGMOD, 2014.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import TIMAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = TIMAlgorithm(graph, model='IC')
        >>> seeds = algo.run(k=10)
    """
    
    def __init__(self, graph: IMGraph, model: str, epsilon: float = 0.5,
                 l: int = 1, random_seed: Optional[int] = None,
                 verbose: bool = False) -> None:
        """初始化 TIM 算法。

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
        """执行 TIM 算法选择种子节点。

        TIM 分两阶段执行，提供 (1-1/e-ε) 近似保证。

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


class TIMPlusAlgorithm:
    """TIM+ 改进算法。

    TIM+ 是 TIM 的改进版本，在第二阶段使用更高效的采样策略。
    相比 TIM，TIM+ 在某些情况下可以减少采样数量。

    该算法同样提供 (1-1/e-ε) 近似保证。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        model: 扩散模型名称（'IC' 或 'LT'）。

    References:
        Youze Tang, Xiaokui Xiao, Yanchen Shi, 
        "Influence Maximization: Near-Optimal Time Complexity Meets Practical Efficiency," 
        in Proc. ACM SIGMOD, 2014.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import TIMPlusAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = TIMPlusAlgorithm(graph, model='IC')
        >>> seeds = algo.run(k=10)
    """
    
    def __init__(self, graph: IMGraph, model: str, epsilon: float = 0.5,
                 l: int = 1, random_seed: Optional[int] = None,
                 verbose: bool = False) -> None:
        """初始化 TIM+ 算法。

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
        """执行 TIM+ 算法选择种子节点。

        TIM+ 分两阶段执行，提供 (1-1/e-ε) 近似保证。

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
