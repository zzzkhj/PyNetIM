"""种群优化算法基类模块。"""

from __future__ import annotations

from typing import Set, TYPE_CHECKING

from ..base_algorithm import BaseAlgorithm

if TYPE_CHECKING:
    from ...graph import IMGraph


class BasePopulationAlgorithm(BaseAlgorithm):
    """种群优化算法基类。

    为基于种群进化的影响力最大化算法提供基础框架，
    包括种群管理、适应度评估、进化操作等功能。

    适用算法：
        - 遗传算法 (Genetic Algorithm)
        - 灰狼优化 (Grey Wolf Optimizer)
        - 粒子群优化 (Particle Swarm Optimization)
        - 蚁群算法 (Ant Colony Optimization)

    Attributes:
        pop_size: 种群大小。
        max_iter: 最大迭代次数。
        mc_rounds: 蒙特卡洛模拟次数（用于适应度评估）。
        population: 种群个体列表。
        population_fitness: 种群适应度列表。

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import BasePopulationAlgorithm
        >>>
        >>> class MyGA(BasePopulationAlgorithm):
        ...     def run(self, k):
        ...         self._init_population(k)
        ...         for _ in range(self.max_iter):
        ...             self._evolution_step(k)
        ...         return self.best_individual
        ...
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = MyGA(graph, pop_size=20, max_iter=50, diffusion_model='IC')
        >>> seeds = algo.run(k=10)
    """

    def __init__(
        self,
        graph: 'IMGraph',
        pop_size: int = 20,
        max_iter: int = 50,
        diffusion_model: str = 'IC',
        mc_rounds: int = 100,
        **kwargs
    ):
        """初始化种群优化算法基类。

        Args:
            graph: 输入图对象。
            pop_size: 种群大小，默认 20。
            max_iter: 最大迭代次数，默认 50。
            diffusion_model: 扩散模型名称，支持 'IC' 或 'LT'，默认 'IC'。
            mc_rounds: 蒙特卡洛模拟次数，默认 100。
            **kwargs: 传递给父类的其他参数。
        """
        super().__init__(graph, diffusion_model, **kwargs)
        
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.mc_rounds = mc_rounds
        
        self.population = None
        self.population_fitness = None
        self.best_individual = None
        self.best_fitness = -1

    def run(self, k: int) -> Set[int]:
        """执行算法选择种子节点。

        子类必须实现此方法来定义具体的进化逻辑。

        Args:
            k: 需要选择的种子节点数量。

        Returns:
            Set[int]: 选中的种子节点集合。

        Raises:
            NotImplementedError: 子类未实现此方法时抛出。
        """
        raise NotImplementedError

    def _init_population(self, k: int):
        """初始化种群。

        子类必须实现此方法来定义种群初始化逻辑。

        Args:
            k: 种子节点数量。

        Raises:
            NotImplementedError: 子类未实现此方法时抛出。
        """
        raise NotImplementedError

    def _evaluate_fitness(self, individual) -> float:
        """评估个体适应度。

        使用扩散模型进行蒙特卡洛模拟来评估个体（种子集合）的影响力传播。

        Args:
            individual: 个体（种子节点集合）。

        Returns:
            float: 适应度值（影响力传播范围）。
        """
        if not individual or len(individual) == 0:
            return 0.0
        
        model = self.diffusion_model(self.graph, individual)
        spread = model.run_monte_carlo_diffusion(self.mc_rounds)
        return spread

    def _evolution_step(self, k: int):
        """执行一步进化。

        子类必须实现此方法来定义进化操作逻辑。

        Args:
            k: 种子节点数量。

        Raises:
            NotImplementedError: 子类未实现此方法时抛出。
        """
        raise NotImplementedError

    def _check_convergence(self) -> bool:
        """检查是否收敛。

        子类可以重写此方法来定义收敛条件。

        Returns:
            bool: 是否收敛。
        """
        return False

    def _update_best(self, individual, fitness: float):
        """更新最佳个体。

        Args:
            individual: 个体。
            fitness: 适应度值。
        """
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_individual = individual
