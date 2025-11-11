from __future__ import annotations

from ..diffusion_model import *
from ..graph import IMGraph


class BaseAlgorithm:
    """
    算法基类。

    该类为各种影响力最大化算法实现提供了基础框架，包含图结构和扩散模型，
    以及需要子类实现的核心方法。

    Attributes:
        graph (IMGraph): 输入图对象
        seeds (list): 种子节点集合
        diffusion_model (BaseDiffusionModel): 扩散模型实例
    """

    def __init__(self, graph: IMGraph, diffusion_model: str | BaseDiffusionModel = None):
        """
        初始化算法基类。

        Args:
            graph (IMGraph): 输入图对象
            diffusion_model (str | BaseDiffusionModel, optional): 扩散模型，支持'IC'或'LT'字符串，
                                                                 或直接传入扩散模型实例，默认为None
        """
        self.graph = graph
        self.seeds = []

        if diffusion_model is not None:
            self.diffusion_model = IndependentCascadeModel
            if isinstance(diffusion_model, str):
                if diffusion_model == 'IC':
                    self.diffusion_model = IndependentCascadeModel
                elif diffusion_model == 'LT':
                    self.diffusion_model = LinearThresholdModel
                else:
                    raise ValueError("不支持的模型：请选择 'IC' 或 'LT'")
            else:
                self.diffusion_model = diffusion_model

    def run(self, k):
        """
        执行算法的抽象方法。

        子类必须实现此方法来定义具体的算法逻辑。

        Args:
            k: 需要选择的种子节点数量

        Returns:
            算法执行结果，具体类型由子类定义

        Raises:
            NotImplementedError: 当子类未实现此方法时抛出
        """
        raise NotImplementedError
