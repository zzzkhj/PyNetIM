from typing import Set, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import IMGraph

from .py_diffusion_model_base import PyDiffusionModelBase


class BaseCallbackDiffusionModel(PyDiffusionModelBase):
    """C++回调版自定义传播模型基类。

    继承 C++ 的 PyDiffusionModelBase，通过回调机制实现自定义传播逻辑。
    单线程模式下可用，比纯 Python 实现快。
    多线程模式受 Python GIL 限制，无法真正并行。

    Attributes:
        graph: 图对象。
        seeds: 种子节点集合。
        num_nodes: 节点数量。

    Example:
        >>> from pynetim.diffusion_model import BaseCallbackDiffusionModel
        >>> 
        >>> class MyICModel(BaseCallbackDiffusionModel):
        ...     def run_single_trial(self, seeds, random_seed):
        ...         import random
        ...         random.seed(random_seed)
        ...         activated = set(seeds)
        ...         current = list(seeds)
        ...         count = len(seeds)
        ...         frequency = [0] * self.graph.num_nodes
        ...         
        ...         while current:
        ...             new_active = []
        ...             for node in current:
        ...                 for neighbor, weight in self.graph.out_neighbors(node):
        ...                     if neighbor not in activated and random.random() < weight:
        ...                         activated.add(neighbor)
        ...                         new_active.append(neighbor)
        ...                         count += 1
        ...                         frequency[neighbor] += 1
        ...             current = new_active
        ...         
        ...         return count, activated, frequency
        ...
        >>> model = MyICModel(graph, {0, 1})
        >>> avg = model.run_monte_carlo_diffusion(mc_rounds=1000, random_seed=42)
    """
    
    def run_single_trial(
        self,
        seeds: List[int],
        random_seed: int
    ) -> Tuple[int, Set[int], List[int]]:
        """执行单次传播试验。

        子类必须重写此方法实现自定义传播逻辑。

        Args:
            seeds: 初始种子节点列表。
            random_seed: 随机数种子，用于确保结果可重现。

        Returns:
            Tuple[int, Set[int], List[int]]: 包含三个元素的元组：
                - 激活节点总数
                - 激活的节点集合
                - 每个节点的激活频数列表

        Raises:
            NotImplementedError: 子类未实现此方法时抛出。

        Note:
            此方法会被 C++ 层回调，因此需要保持签名一致。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 必须实现 run_single_trial(seeds, random_seed) 方法"
        )
