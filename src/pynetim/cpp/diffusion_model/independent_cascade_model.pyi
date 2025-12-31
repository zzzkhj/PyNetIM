from typing import Set
from ..graph.graph import IMGraphCpp


class IndependentCascadeModel:
    """
    Independent Cascade (IC) diffusion model.

    本类实现经典的 Independent Cascade 传播模型，
    用于模拟信息 / 影响力在图结构中的随机扩散过程。

    具体传播逻辑由 C++ 后端实现，
    本文件仅用于 Python 侧的类型提示与文档说明。
    """

    # ================== 构造函数 ==================

    def __init__(self, seeds: Set[int], graph: IMGraphCpp) -> None:
        """
        初始化 Independent Cascade 模型

        Args
        ----------
        seeds : Set[int]
            初始种子节点集合
        graph : Graph
            底层图结构（只读，不应在扩散过程中被修改）
        """
        ...

    # ================== 状态更新 ==================

    def set_seeds(self, seeds: Set[int]) -> None:
        """
        更新种子节点集合

        Args
        ----------
        seeds : Set[int]
            新的初始激活节点集合
        """
        ...

    # ================== 扩散模拟 ==================

    def run_monte_carlo_diffusion(
        self,
        rounds: int,
        seed: int = ...,
        use_multithread: bool = ...
    ) -> float:
        """
        运行 Monte Carlo 扩散模拟

        在给定初始种子节点的情况下，
        重复进行多轮随机扩散实验，
        统计最终被激活节点数的期望值。

        Args
        ----------
        rounds : int
            Monte Carlo 模拟轮数
        seed : int, optional
            随机数种子（用于保证实验可复现性）
        use_multithread : bool, optional
            是否启用多线程加速模拟

        Returns
        -------
        float
            扩散结束后激活节点数量的期望值
        """
        ...
