"""强化学习影响力最大化算法基类模块。"""

from __future__ import annotations

from typing import Set, TYPE_CHECKING

from ..base_algorithm import BaseAlgorithm

if TYPE_CHECKING:
    from ...graph import IMGraph


class BaseRLAlgorithm(BaseAlgorithm):
    """强化学习影响力最大化算法基类。

    为基于强化学习的 IM 算法提供基础框架，
    包括状态管理、动作选择、奖励计算等功能。

    适用算法：
        - Q-learning
        - Deep Q-Network (DQN)
        - Policy Gradient

    Attributes:
        alpha: 学习率。
        gamma: 折扣因子。
        epsilon: 探索率。

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import BaseRLAlgorithm
        >>>
        >>> class MyQLearning(BaseRLAlgorithm):
        ...     def run(self, k):
        ...         self._init_q_table()
        ...         for episode in range(self.episodes):
        ...             self._train_episode(k)
        ...         return self._select_seeds(k)
        ...
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = MyQLearning(graph, alpha=0.1, gamma=0.9, epsilon=0.5)
        >>> seeds = algo.run(k=10)
    """

    def __init__(
        self,
        graph: 'IMGraph',
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.5,
        diffusion_model: str = 'IC',
        mc_rounds: int = 100,
        **kwargs
    ):
        """初始化强化学习算法基类。

        Args:
            graph: 输入图对象。
            alpha: 学习率，默认 0.1。
            gamma: 折扣因子，默认 0.9。
            epsilon: 探索率，默认 0.5。
            diffusion_model: 扩散模型名称，支持 'IC' 或 'LT'，默认 'IC'。
            mc_rounds: 蒙特卡洛模拟次数，默认 100。
            **kwargs: 传递给父类的其他参数。
        """
        super().__init__(graph, diffusion_model, **kwargs)
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.mc_rounds = mc_rounds
        
        self.q_table = None
        self.candidate_set = None

    def run(self, k: int) -> Set[int]:
        """执行算法选择种子节点。

        子类必须实现此方法来定义具体的强化学习逻辑。

        Args:
            k: 需要选择的种子节点数量。

        Returns:
            Set[int]: 选中的种子节点集合。

        Raises:
            NotImplementedError: 子类未实现此方法时抛出。
        """
        raise NotImplementedError

    def _init_q_table(self):
        """初始化 Q 表。

        子类必须实现此方法来定义 Q 表初始化逻辑。

        Raises:
            NotImplementedError: 子类未实现此方法时抛出。
        """
        raise NotImplementedError

    def _select_action(self, state, visited) -> int:
        """选择动作（节点）。

        基于 epsilon-greedy 策略选择动作。

        Args:
            state: 当前状态。
            visited: 已访问的节点集合。

        Returns:
            int: 选择的动作（节点索引）。
        """
        raise NotImplementedError

    def _update_q_value(self, state, action, reward, next_state):
        """更新 Q 值。

        使用 Q-learning 更新规则：
        Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))

        Args:
            state: 当前状态。
            action: 执行的动作。
            reward: 获得的奖励。
            next_state: 下一状态。
        """
        raise NotImplementedError
