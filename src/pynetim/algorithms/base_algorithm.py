from __future__ import annotations

from typing import TYPE_CHECKING, Set

if TYPE_CHECKING:
    from ..graph import IMGraph
    from ..diffusion_model import IndependentCascadeModel, LinearThresholdModel


class BaseAlgorithm:
    """影响力最大化算法基类。

    为各种影响力最大化算法实现提供基础框架，包含图结构和扩散模型，
    以及需要子类实现的核心方法。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。
        diffusion_model: 扩散模型类。

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import BaseAlgorithm
        >>> 
        >>> class MyAlgorithm(BaseAlgorithm):
        ...     def run(self, k):
        ...         return set(range(k))
        ...
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = MyAlgorithm(graph, diffusion_model='IC')
        >>> seeds = algo.run(k=10)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = None):
        """初始化算法基类。

        Args:
            graph: 输入图对象。
            diffusion_model: 扩散模型名称，支持 'IC' 或 'LT'，默认为 None。

        Raises:
            ValueError: 当 diffusion_model 不是 'IC' 或 'LT' 时抛出。
        """
        self.graph = graph
        self.seeds: Set[int] = set()

        if diffusion_model is not None:
            from ..diffusion_model import IndependentCascadeModel, LinearThresholdModel
            
            if diffusion_model == 'IC':
                self.diffusion_model = IndependentCascadeModel
            elif diffusion_model == 'LT':
                self.diffusion_model = LinearThresholdModel
            else:
                raise ValueError("不支持的模型：请选择 'IC' 或 'LT'")
        else:
            self.diffusion_model = None

    def run(self, k: int) -> Set[int]:
        """执行算法选择种子节点。

        子类必须实现此方法来定义具体的算法逻辑。

        Args:
            k: 需要选择的种子节点数量。

        Returns:
            Set[int]: 选中的种子节点集合。

        Raises:
            NotImplementedError: 子类未实现此方法时抛出。
        """
        raise NotImplementedError

    def get_seeds(self) -> Set[int]:
        """获取最后一次运行选出的种子集合。

        Returns:
            Set[int]: 种子节点集合。
        """
        return self.seeds
