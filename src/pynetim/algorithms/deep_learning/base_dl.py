from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import torch

from ..base_algorithm import BaseAlgorithm

if TYPE_CHECKING:
    from ...graph import IMGraph


class BaseDLAlgorithm(BaseAlgorithm):
    """深度学习影响力最大化算法基类。

    为深度学习算法提供基础框架，包含设备管理、权重加载等功能。

    Attributes:
        device: 计算设备 (CPU/GPU)。
        model: 神经网络模型。

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import ToupleGDDAlgorithm
        >>>
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = ToupleGDDAlgorithm(graph, pretrained=True)
        >>> seeds = algo.run(k=10)
    """

    _weights_filename: str = None

    def __init__(
        self,
        graph: 'IMGraph',
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        device: str = 'auto'
    ):
        """初始化深度学习算法基类。

        Args:
            graph: 输入图对象。
            pretrained: 是否使用预训练权重，默认为 True。
            weights_path: 本地权重路径，优先级高于 pretrained。
            device: 计算设备，支持 'auto'、'cpu'、'cuda'，默认为 'auto'。
        """
        super().__init__(graph)
        self.device = self._get_device(device)
        self._node_embed = None
        self.model = None

    def _get_device(self, device: str) -> torch.device:
        """获取计算设备。

        Args:
            device: 设备字符串，支持 'auto'、'cpu'、'cuda'。

        Returns:
            torch.device: 计算设备对象。
        """
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _load_weights(self, weights_path: Optional[str] = None):
        """加载模型权重。

        Args:
            weights_path: 权重文件路径。

        Raises:
            FileNotFoundError: 权重文件不存在时抛出。
            NotImplementedError: 子类未实现此方法时抛出。
        """
        raise NotImplementedError

    def _get_weights_path(self, weights_path: Optional[str] = None) -> str:
        """获取权重文件路径。

        Args:
            weights_path: 用户指定的权重路径。

        Returns:
            str: 权重文件路径。

        Raises:
            NotImplementedError: 子类未设置 _weights_filename 时抛出。
        """
        if weights_path is not None:
            return weights_path
        if self._weights_filename is None:
            raise NotImplementedError(
                "子类必须设置 _weights_filename 属性或重写 _get_weights_path 方法"
            )
        return os.path.join(os.path.dirname(__file__), 'touplegdd', 'weights', self._weights_filename)
