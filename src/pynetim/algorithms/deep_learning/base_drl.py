from __future__ import annotations

from typing import Set, TYPE_CHECKING

import torch

from .base_dl import BaseDLAlgorithm

if TYPE_CHECKING:
    from ...graph import IMGraph


class BaseDRLAlgorithm(BaseDLAlgorithm):
    """深度强化学习影响力最大化算法基类。

    为基于 DQN 的算法提供通用的种子选择框架。

    子类需要实现：
        - _prepare_inference(): 准备推理环境
        - _init_state(): 初始化状态
        - _compute_q_values(state): 计算 Q 值
        - _update_state(state, node): 更新状态
        - _mask_selected(q_values, state): 屏蔽已选节点

    Example:
        >>> class MyDRLAlgo(BaseDRLAlgorithm):
        ...     def _init_state(self):
        ...         return torch.zeros(self.graph.num_nodes)
        ...     def _compute_q_values(self, state):
        ...         return self.model(state)
        ...     # ... 其他方法
    """

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
