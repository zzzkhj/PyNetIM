from typing import Set
from ..graph.graph import IMGraphCpp


class LinearThresholdModel:
    """
    Linear Threshold (LT) diffusion model with randomized activation thresholds.

    本模型是经典 Linear Threshold 模型的扩展版本。
    每个节点的激活阈值 θ 从区间 [θ_l, θ_h] 中随机采样，
    当来自已激活邻居的累积影响不小于该阈值时，节点被激活。
    """

    # ================== 构造函数 ==================

    def __init__(
        self,
        seeds: Set[int],
        graph: IMGraphCpp,
        theta_l: float,
        theta_h: float
    ) -> None:
        """
        初始化 Linear Threshold 扩散模型（带阈值区间）

        Args
        ----------
        seeds : Set[int]
            初始种子节点集合
        graph : Graph
            底层图结构（只读）
        theta_l : float
            激活阈值下界
        theta_h : float
            激活阈值上界
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
        运行 Monte Carlo 扩散模拟（LT 模型）

        在每一轮模拟中：
        - 为每个节点从区间 [θ_l, θ_h] 中随机采样激活阈值
        - 节点累积其所有已激活入邻居的影响权重
        - 当累积影响 ≥ 节点阈值时，该节点被激活
        - 扩散过程持续直到无新节点被激活

        Args
        ----------
        rounds : int
            Monte Carlo 模拟轮数
        seed : int, optional
            随机数种子（用于结果可复现）
        use_multithread : bool, optional
            是否启用多线程并行模拟

        Returns
        -------
        float
            扩散结束后激活节点数量的期望值
        """
        ...
