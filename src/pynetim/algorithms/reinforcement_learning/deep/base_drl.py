from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Set, TYPE_CHECKING

import torch

from ..base_rl import BaseRLAlgorithm

if TYPE_CHECKING:
    from ...graph import IMGraph


class BaseDRLAlgorithm(BaseRLAlgorithm):
    """深度强化学习影响力最大化算法基类。

    为基于 DQN 的算法提供通用的种子选择框架。

    子类需要实现：
        - _prepare_inference(): 准备推理环境
        - _init_state(): 初始化状态
        - _compute_q_values(state): 计算 Q 值
        - _update_state(state, node): 更新状态
        - _mask_selected(q_values, state): 屏蔽已选节点

    Attributes:
        device: 计算设备。
        pretrained: 是否使用预训练权重。
        weights_path: 本地权重路径。

    Example:
        >>> class MyDRLAlgo(BaseDRLAlgorithm):
        ...     def _init_state(self):
        ...         return torch.zeros(self.graph.num_nodes)
        ...     def _compute_q_values(self, state):
        ...         return self.model(state)
        ...     # ... 其他方法
    """

    _weights_filename: str = None

    def __init__(
        self,
        graph: 'IMGraph',
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        device: str = 'auto',
        diffusion_model: str = 'IC',
        mc_rounds: int = 100,
        **kwargs
    ):
        """初始化深度强化学习算法基类。

        Args:
            graph: 输入图对象。
            pretrained: 是否使用预训练权重，默认为 True。
            weights_path: 本地权重路径，优先级高于 pretrained。
            device: 计算设备，支持 'auto'、'cpu'、'cuda'，默认为 'auto'。
            diffusion_model: 扩散模型名称，支持 'IC' 或 'LT'，默认为 'IC'。
            mc_rounds: 蒙特卡洛模拟次数，默认为 100。
            **kwargs: 传递给父类的其他参数。
        """
        super().__init__(graph, diffusion_model=diffusion_model, mc_rounds=mc_rounds, **kwargs)
        
        self.device = self._get_device(device)
        self.pretrained = pretrained
        self.weights_path = weights_path
        self._node_embed = None

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

    def _get_weights_path(self, weights_path: Optional[str] = None) -> Path:
        """获取权重文件路径。

        Args:
            weights_path: 用户指定的权重路径。

        Returns:
            Path: 权重文件路径。
        """
        if weights_path is not None:
            return Path(weights_path)
        
        if self._weights_filename is None:
            raise ValueError("子类必须定义 _weights_filename")
        
        return Path(__file__).parent / 'touplegdd' / 'weights' / self._weights_filename

    @torch.no_grad()
    def run(self, k: int, use_topk: bool = False) -> Set[int]:
        """执行算法选择种子节点。

        Args:
            k: 种子节点数量。
            use_topk: 是否使用 top-k 选择策略。
                - True: 一次性选择 top-k 节点。
                - False: 迭代选择，每次选择一个节点后更新状态。

        Returns:
            Set[int]: 选择的种子节点集合。
        """
        self._prepare_inference()
        state = self._init_state()

        if use_topk:
            self.seeds = self._select_topk(k, state)
        else:
            self.seeds = self._select_iterative(k, state)

        return self.seeds

    def _prepare_inference(self):
        """准备推理环境（子类可重写）。"""
        pass

    def _init_state(self):
        """初始化状态。

        Returns:
            状态对象（具体类型由子类决定）。
        """
        raise NotImplementedError("子类必须实现 _init_state 方法")

    def _compute_q_values(self, state) -> torch.Tensor:
        """计算 Q 值。

        Args:
            state: 当前状态。

        Returns:
            torch.Tensor: 各节点的 Q 值。
        """
        raise NotImplementedError("子类必须实现 _compute_q_values 方法")

    def _update_state(self, state, node: int):
        """更新状态。

        Args:
            state: 当前状态。
            node: 新选择的节点。

        Returns:
            更新后的状态。
        """
        raise NotImplementedError("子类必须实现 _update_state 方法")

    def _mask_selected(self, q_values: torch.Tensor, state) -> torch.Tensor:
        """屏蔽已选节点的 Q 值。

        Args:
            q_values: Q 值张量。
            state: 当前状态。

        Returns:
            torch.Tensor: 屏蔽后的 Q 值。
        """
        raise NotImplementedError("子类必须实现 _mask_selected 方法")

    def _select_topk(self, k: int, state) -> Set[int]:
        """一次性选择 top-k 节点。

        Args:
            k: 种子节点数量。
            state: 初始状态。

        Returns:
            Set[int]: 选择的种子节点集合。
        """
        q_values = self._compute_q_values(state)
        _, indices = torch.topk(q_values, k)
        return set(indices.tolist())

    def _select_iterative(self, k: int, state) -> Set[int]:
        """迭代选择节点。

        Args:
            k: 种子节点数量。
            state: 初始状态。

        Returns:
            Set[int]: 选择的种子节点集合。
        """
        selected = set()

        for _ in range(k):
            q_values = self._compute_q_values(state)
            q_values = self._mask_selected(q_values, state)

            best_node = torch.argmax(q_values).item()
            selected.add(best_node)
            state = self._update_state(state, best_node)

        return selected
