"""RIS算法模块，提供基于反向影响采样的影响力最大化算法"""

from typing import Set, Optional
from ..graph import IMGraph

class BaseRISAlgorithm:
    """反向影响采样算法基类。

    基于反向影响采样（Reverse Influence Sampling, RIS）的影响力最大化算法。
    用户指定 RR 集合数量，使用堆优化贪心选择。

    该算法速度快且可扩展性好，适合大规模图。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        model: 扩散模型名称（'IC' 或 'LT'）。

    References:
        Borgs, C., Brautbar, M., Chitnis, N., & Tardos, É. (2014). 
        Maximizing social influence in nearly optimal time. SODA, 946-957.

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
        """初始化 RIS 算法。

        Args:
            graph: 输入图对象。
            model: 扩散模型名称，支持 'IC' 或 'LT'。
            random_seed: 随机种子，默认为 None（每次随机）。
            verbose: 是否显示关键过程日志，默认为 False。
        """
        ...
    
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
        ...
    
    def get_seeds(self) -> Set[int]:
        """获取最后一次运行选出的种子集合。

        Returns:
            Set[int]: 种子节点集合。
        """
        ...


class IMMAlgorithm(BaseRISAlgorithm):
    """基于鞅的影响力最大化算法。

    IMM (Influence Maximization via Martingales) 是 RIS 算法的改进版本，
    自动估计最优采样数量，在保证理论精度的同时提高效率。

    该算法速度快、精度高、可扩展性好，适合大规模图。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        model: 扩散模型名称（'IC' 或 'LT'）。
        epsilon: 近似参数 ε。
        l: 失败概率参数。

    References:
        Tang, Y., Xiao, X., & Shi, Y. (2015). Influence maximization: 
        Near-optimal time complexity meets practical efficiency. SIGMOD, 75-86.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import IMMAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = IMMAlgorithm(graph, model='IC', epsilon=0.5)
        >>> seeds = algo.run(k=10)
    """
    
    def __init__(self, graph: IMGraph, model: str, epsilon: float = 0.5, 
                 l: int = 1, random_seed: Optional[int] = None, 
                 verbose: bool = False) -> None:
        """初始化 IMM 算法。

        Args:
            graph: 输入图对象。
            model: 扩散模型名称，支持 'IC' 或 'LT'。
            epsilon: 近似参数ε，默认为 0.5。
            l: 失败概率参数，默认为 1。
            random_seed: 随机种子，默认为 None（每次随机）。
            verbose: 是否显示关键过程日志，默认为 False。
        """
        ...
    
    def run(self, k: int) -> Set[int]:
        """执行 IMM 算法选择种子节点。

        IMM会自动估计最优采样数量，无需手动指定。

        Args:
            k: 需要选择的种子节点数量。

        Returns:
            Set[int]: 选中的种子节点集合。
        """
        ...


class TIMAlgorithm(BaseRISAlgorithm):
    """两阶段影响力最大化算法。

    TIM (Two-phase Influence Maximization) 是 RIS 算法的改进版本，
    使用两阶段策略：第一阶段估计OPT，第二阶段采样并选择种子。

    该算法理论保证好，适合大规模图。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        model: 扩散模型名称（'IC' 或 'LT'）。
        epsilon: 近似参数 ε。
        l: 失败概率参数。

    References:
        Tang, Y., Xiao, X., & Shi, Y. (2014). Influence maximization: 
        Near-optimal time complexity meets practical efficiency. SIGMOD, 75-86.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import TIMAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = TIMAlgorithm(graph, model='IC', epsilon=0.5)
        >>> seeds = algo.run(k=10)
    """
    
    def __init__(self, graph: IMGraph, model: str, epsilon: float = 0.5, 
                 l: int = 1, random_seed: Optional[int] = None, 
                 verbose: bool = False) -> None:
        """初始化 TIM 算法。

        Args:
            graph: 输入图对象。
            model: 扩散模型名称，支持 'IC' 或 'LT'。
            epsilon: 近似参数ε，默认为 0.5。
            l: 失败概率参数，默认为 1。
            random_seed: 随机种子，默认为 None（每次随机）。
            verbose: 是否显示关键过程日志，默认为 False。
        """
        ...
    
    def run(self, k: int) -> Set[int]:
        """执行 TIM 算法选择种子节点。

        TIM使用两阶段策略：第一阶段估计OPT，第二阶段采样并选择种子。

        Args:
            k: 需要选择的种子节点数量。

        Returns:
            Set[int]: 选中的种子节点集合。
        """
        ...


class TIMPlusAlgorithm(BaseRISAlgorithm):
    """改进的两阶段影响力最大化算法。

    TIM+ 是 TIM 的改进版本，使用更高效的采样策略，
    通常比 TIM 更快。

    该算法速度快、理论保证好，适合大规模图。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        model: 扩散模型名称（'IC' 或 'LT'）。
        epsilon: 近似参数 ε。
        l: 失败概率参数。

    References:
        Tang, Y., Xiao, X., & Shi, Y. (2014). Influence maximization: 
        Near-optimal time complexity meets practical efficiency. SIGMOD, 75-86.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import TIMPlusAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = TIMPlusAlgorithm(graph, model='IC', epsilon=0.5)
        >>> seeds = algo.run(k=10)
    """
    
    def __init__(self, graph: IMGraph, model: str, epsilon: float = 0.5, 
                 l: int = 1, random_seed: Optional[int] = None, 
                 verbose: bool = False) -> None:
        """初始化 TIM+ 算法。

        Args:
            graph: 输入图对象。
            model: 扩散模型名称，支持 'IC' 或 'LT'。
            epsilon: 近似参数ε，默认为 0.5。
            l: 失败概率参数，默认为 1。
            random_seed: 随机种子，默认为 None（每次随机）。
            verbose: 是否显示关键过程日志，默认为 False。
        """
        ...
    
    def run(self, k: int) -> Set[int]:
        """执行 TIM+ 算法选择种子节点。

        TIM+改进了TIM的采样策略，通常比TIM更快。

        Args:
            k: 需要选择的种子节点数量。

        Returns:
            Set[int]: 选中的种子节点集合。
        """
        ...
