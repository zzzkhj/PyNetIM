from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set, TYPE_CHECKING

import torch

from ..base_drl import BaseDRLAlgorithm
from .agent import get_q_net_input

if TYPE_CHECKING:
    from ....graph import IMGraph


class BiGDNBaseAlgorithm(BaseDRLAlgorithm):
    """BiGDN 系列算法基类。

    为 BiGDN 和 BiGDNS 提供通用的种子选择逻辑。

    Attributes:
        num_features: 特征维度。
        verbose: 是否输出详细信息。
        agent: 智能体对象。
    """

    _weights_filename: str = None

    def __init__(
        self,
        graph: 'IMGraph',
        num_features: int = 64,
        verbose: bool = False,
        **kwargs
    ):
        """初始化 BiGDN 基类。

        Args:
            graph: 输入图对象。
            num_features: 特征维度，默认为 64。
            verbose: 是否输出详细信息，默认为 False。
            **kwargs: 传递给父类的其他参数。
        """
        super().__init__(graph, **kwargs)
        self.num_features = num_features
        self.verbose = verbose

    def _prepare_inference(self):
        """准备推理环境。"""
        self.agent.q_net.eval()

    def _init_state(self) -> List[int]:
        """初始化状态。

        Returns:
            List[int]: 初始状态向量（全 0）。
        """
        return [0] * self.graph.num_nodes

    def _compute_q_values(self, state: List[int]) -> torch.Tensor:
        """计算 Q 值。

        Args:
            state: 当前状态向量。

        Returns:
            torch.Tensor: 各节点的 Q 值。
        """
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
        data = get_q_net_input([self.graph], self.num_features, self.device)
        return self.agent.q_net(data.x, data.edge_index, data.edge_weight, data.batch, state_tensor)

    def _update_state(self, state: List[int], node: int) -> List[int]:
        """更新状态。

        Args:
            state: 当前状态向量。
            node: 新选择的节点。

        Returns:
            List[int]: 更新后的状态向量。
        """
        state[node] = 1
        return state

    def _mask_selected(self, q_values: torch.Tensor, state: List[int]) -> torch.Tensor:
        """屏蔽已选节点的 Q 值。

        Args:
            q_values: Q 值张量。
            state: 当前状态向量。

        Returns:
            torch.Tensor: 屏蔽后的 Q 值。
        """
        for i, s in enumerate(state):
            if s == 1:
                q_values[i] = -1e10
        return q_values

    def _select_iterative(self, k: int, state) -> Set[int]:
        """迭代选择节点（带 verbose 输出）。

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

            if self.verbose:
                print(f"Selected node {best_node}, Q-value: {q_values[best_node].item():.4f}")

        return selected

    def _get_weights_path(self) -> Path:
        """获取权重文件路径。

        Returns:
            Path: 权重文件路径。
        """
        return Path(__file__).parent / "weights" / self._weights_filename

    def _load_weights(self) -> bool:
        """加载预训练权重。

        Returns:
            bool: 是否成功加载。
        """
        weights_path = self._get_weights_path()
        if weights_path.exists():
            self.agent.q_net.load_state_dict(
                torch.load(weights_path, map_location=self.device, weights_only=True)
            )
            self.agent.target_q_net.load_state_dict(self.agent.q_net.state_dict())
            if self.verbose:
                print(f"Loaded weights from {weights_path}")
            return True
        return False

    def _load_weights_from_path(self, weights_path: str) -> bool:
        """从指定路径加载权重。

        Args:
            weights_path: 权重文件路径。

        Returns:
            bool: 是否成功加载。
        """
        path = Path(weights_path)
        if path.exists():
            self.agent.q_net.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            self.agent.target_q_net.load_state_dict(self.agent.q_net.state_dict())
            if self.verbose:
                print(f"Loaded weights from {path}")
            return True
        return False
