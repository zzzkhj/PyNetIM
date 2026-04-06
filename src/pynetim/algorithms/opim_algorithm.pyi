"""OPIM算法模块，提供Online Processing for Influence Maximization算法"""

from typing import Set, Optional
from ..graph import IMGraph

class OPIMAlgorithm:
    """Online Processing for Influence Maximization 算法。

    OPIM 使用两组独立的 RR 集合：
    - R1 用于贪心选择种子节点
    - R2 用于验证选中种子的影响力

    该算法提供可证明的近似保证，适合需要理论保证的场景。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        model: 扩散模型名称（'IC' 或 'LT'）。

    References:
        Jing Tang, Xueyan Tang, Xiaokui Xiao, Junsong Yuan, 
        "Online Processing Algorithms for Influence Maximization," 
        in Proc. ACM SIGMOD, 2018.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import OPIMAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = OPIMAlgorithm(graph, model='IC')
        >>> seeds = algo.run(k=10, num_rr_sets=10000)
        >>> approx = algo.get_approximation()
    """
    
    def __init__(self, graph: IMGraph, model: str, random_seed: Optional[int] = None, 
                 verbose: bool = False) -> None:
        """初始化 OPIM 算法。

        Args:
            graph: 输入图对象。
            model: 扩散模型名称，支持 'IC' 或 'LT'。
            random_seed: 随机种子，默认为 None（每次随机）。
            verbose: 是否显示关键过程日志，默认为 False。
        """
        ...
    
    def run(self, k: int, num_rr_sets: int, delta: float = -1.0, mode: int = 2) -> Set[int]:
        """执行 OPIM 算法选择种子节点。

        OPIM 使用固定数量的 RR 集合，最大化近似保证。

        Args:
            k: 需要选择的种子节点数量。
            num_rr_sets: RR 集合总采样数量（会均分给 R1 和 R2）。
            delta: 失败概率参数，默认为 1/n。
            mode: 上界计算模式：
                - 0: Vanilla 版本，返回 (1-1/e)-近似
                - 1: 使用最后一轮的上界
                - 2: 使用所有轮中的最小上界（默认）

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
    
    def get_approximation(self) -> float:
        """获取最后一次运行的近似保证值。

        Returns:
            float: 近似保证值 α。
        """
        ...
    
    def get_influence(self) -> float:
        """获取最后一次运行的影响力估计值（通过 R2 验证集）。

        Returns:
            float: 影响力估计值。
        """
        ...


class OPIMCAlgorithm(OPIMAlgorithm):
    """OPIM-C 自适应采样算法。

    OPIM-C 是 OPIM 的自适应版本，迭代增加 RR 集合数量，
    直到达到 (1-1/e-ε) 近似保证。

    该算法自动确定采样数量，适合需要特定精度保证的场景。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        model: 扩散模型名称（'IC' 或 'LT'）。

    References:
        Jing Tang, Xueyan Tang, Xiaokui Xiao, Junsong Yuan, 
        "Online Processing Algorithms for Influence Maximization," 
        in Proc. ACM SIGMOD, 2018.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import OPIMCAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = OPIMCAlgorithm(graph, model='IC')
        >>> seeds = algo.run(k=10, epsilon=0.1)
    """
    
    def __init__(self, graph: IMGraph, model: str, random_seed: Optional[int] = None, 
                 verbose: bool = False) -> None:
        """初始化 OPIM-C 算法。

        Args:
            graph: 输入图对象。
            model: 扩散模型名称，支持 'IC' 或 'LT'。
            random_seed: 随机种子，默认为 None（每次随机）。
            verbose: 是否显示关键过程日志，默认为 False。
        """
        ...
    
    def run(self, k: int, epsilon: float, delta: float = -1.0, mode: int = 2) -> Set[int]:
        """执行 OPIM-C 算法选择种子节点。

        OPIM-C 会自动迭代增加 RR 集合数量，直到达到目标近似保证。

        Args:
            k: 需要选择的种子节点数量。
            epsilon: 误差阈值，算法返回 (1-1/e-ε)-近似解。
            delta: 失败概率参数，默认为 1/n。
            mode: 上界计算模式：
                - 0: Vanilla 版本
                - 1: 使用最后一轮的上界
                - 2: 使用所有轮中的最小上界（默认）

        Returns:
            Set[int]: 选中的种子节点集合。
        """
        ...
