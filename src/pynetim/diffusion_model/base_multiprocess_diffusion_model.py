import random
import multiprocessing as mp
from typing import Set, List, Tuple, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..graph import IMGraph

_global_graph = None
_global_model_class = None


def _init_worker(graph_data, model_class):
    """初始化工作进程。

    Args:
        graph_data: 图数据字典，包含 edges、weights、directed。
        model_class: 模型类。
    """
    global _global_graph, _global_model_class
    import sys
    sys.path.insert(0, '/root/PyNetIM/src')
    from pynetim import IMGraph
    _global_graph = IMGraph(graph_data['edges'], graph_data['weights'], graph_data['directed'])
    _global_model_class = model_class


def _run_trial_worker(args):
    """执行单次试验的工作函数。

    Args:
        args: 包含 (seeds, random_seed) 的元组。

    Returns:
        单次试验的结果元组。
    """
    seeds, random_seed = args
    model = _global_model_class(_global_graph, seeds)
    return model.run_single_trial(list(seeds), random_seed)


class BaseMultiprocessDiffusionModel(ABC):
    """多进程版自定义传播模型基类。

    纯 Python 实现，使用 multiprocessing 实现真正的并行计算。
    不受 Python GIL 限制，多进程可真正并行加速。
    适合大规模模拟和需要并行加速的自定义模型。

    Attributes:
        graph: 图对象。
        seeds: 种子节点集合。
        num_nodes: 节点数量。

    Example:
        >>> from pynetim.diffusion_model import BaseMultiprocessDiffusionModel
        >>> 
        >>> class MyICModel(BaseMultiprocessDiffusionModel):
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
        ...                 for neighbor, weight in self.graph.out_neighbors_with_weights(node):
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
        >>> avg = model.run_monte_carlo_diffusion(mc_rounds=1000, num_processes=4, random_seed=42)
    """
    
    def __init__(self, graph: 'IMGraph', seeds: Set[int]):
        """初始化传播模型。

        Args:
            graph: 图对象。
            seeds: 种子节点集合。
        """
        self.graph = graph
        self.seeds = set(seeds)
        self.num_nodes = graph.num_nodes
    
    def set_seeds(self, seeds: Set[int]):
        """设置种子节点集合。

        Args:
            seeds: 新的种子节点集合。
        """
        self.seeds = set(seeds)
    
    @abstractmethod
    def run_single_trial(self, seeds: List[int], random_seed: int) -> Tuple[int, Set[int], List[int]]:
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
        """
        raise NotImplementedError
    
    def _get_graph_data(self) -> dict:
        """获取图数据用于进程间传递。

        Returns:
            dict: 包含 edges、weights、directed 的字典。
        """
        sparse_matrix = self.graph.get_adj_matrix_sparse()
        edges = [(u, v) for u, v, _ in sparse_matrix]
        weights = [w for _, _, w in sparse_matrix]
        return {
            'edges': edges,
            'weights': weights,
            'directed': self.graph.directed
        }
    
    def run_monte_carlo_diffusion(
        self, 
        mc_rounds: int, 
        num_processes: int = None,
        show_progress: bool = False,
        random_seed: int = None
    ) -> float:
        """运行蒙特卡洛模拟，计算平均影响力。

        Args:
            mc_rounds: 蒙特卡洛模拟次数，建议 1000-10000 次。
            num_processes: 进程数，默认使用 CPU 核心数。
            show_progress: 是否显示进度条（暂未实现）。
            random_seed: 随机数种子，默认为 None（每次结果不同）。

        Returns:
            float: 平均激活节点数。

        Note:
            当模拟次数较多时（>= 进程数 * 10），自动启用多进程并行。
        """
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        if random_seed is None:
            base_seed = random.randint(0, 2**31 - 1)
        else:
            base_seed = random_seed
        rng_seeds = [base_seed + i for i in range(mc_rounds)]
        
        if num_processes > 1 and mc_rounds >= num_processes * 10:
            graph_data = self._get_graph_data()
            args_list = [(self.seeds, rng_seeds[i]) for i in range(mc_rounds)]
            
            ctx = mp.get_context('fork')
            with ctx.Pool(
                processes=num_processes,
                initializer=_init_worker,
                initargs=(graph_data, self.__class__)
            ) as pool:
                results = pool.map(_run_trial_worker, args_list)
            
            total_activated = sum(r[0] for r in results)
        else:
            total_activated = 0
            for i in range(mc_rounds):
                count, _, _ = self.run_single_trial(list(self.seeds), rng_seeds[i])
                total_activated += count
        
        return total_activated / mc_rounds
    
    def run_monte_carlo_with_frequency(
        self, 
        mc_rounds: int, 
        num_processes: int = None,
        random_seed: int = None
    ) -> Tuple[float, List[int]]:
        """运行蒙特卡洛模拟，返回平均影响力和激活频数。

        Args:
            mc_rounds: 蒙特卡洛模拟次数，建议 1000-10000 次。
            num_processes: 进程数，默认使用 CPU 核心数。
            random_seed: 随机数种子，默认为 None（每次结果不同）。

        Returns:
            Tuple[float, List[int]]: 包含两个元素：
                - 平均激活节点数
                - 每个节点的激活频数列表
        """
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        if random_seed is None:
            base_seed = random.randint(0, 2**31 - 1)
        else:
            base_seed = random_seed
        rng_seeds = [base_seed + i for i in range(mc_rounds)]
        
        if num_processes > 1 and mc_rounds >= num_processes * 10:
            graph_data = self._get_graph_data()
            args_list = [(self.seeds, rng_seeds[i]) for i in range(mc_rounds)]
            
            ctx = mp.get_context('fork')
            with ctx.Pool(
                processes=num_processes,
                initializer=_init_worker,
                initargs=(graph_data, self.__class__)
            ) as pool:
                results = pool.map(_run_trial_worker, args_list)
            
            total_activated = sum(r[0] for r in results)
            total_frequency = [0] * self.num_nodes
            for _, _, freq in results:
                for i, f in enumerate(freq):
                    total_frequency[i] += f
        else:
            total_activated = 0
            total_frequency = [0] * self.num_nodes
            for i in range(mc_rounds):
                count, _, freq = self.run_single_trial(list(self.seeds), rng_seeds[i])
                total_activated += count
                for j, f in enumerate(freq):
                    total_frequency[j] += f
        
        return total_activated / mc_rounds, total_frequency
