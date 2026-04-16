import os
import random
from typing import Set, Optional, TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ..base_dl import BaseDLAlgorithm
from . import models

if TYPE_CHECKING:
    from ...graph import IMGraph

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)


class ToupleGDDAlgorithm(BaseDLAlgorithm):
    """ToupleGDD 深度学习影响力最大化算法。

    使用三重门控图神经网络 (Tripling GNN) 结合强化学习选择种子节点。

    References:
        Chen T, Yan S, Guo J, Wu W. ToupleGDD: A Fine-Designed Solution of Influence
        Maximization by Deep Reinforcement Learning. IEEE Transactions on Computational
        Social Systems, 2024.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import ToupleGDDAlgorithm
        >>>
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = ToupleGDDAlgorithm(graph, pretrained=True)
        >>> seeds = algo.run(k=10)
    """

    _weights_filename = 'tripling.ckpt'

    def __init__(
        self,
        graph: 'IMGraph',
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        device: str = 'auto'
    ):
        """初始化 ToupleGDD 算法。

        Args:
            graph: 输入图对象。
            pretrained: 是否使用预训练权重，默认为 True。
            weights_path: 本地权重路径，优先级高于 pretrained。
            device: 计算设备，支持 'auto'、'cpu'、'cuda'，默认为 'auto'。
        """
        super().__init__(graph, pretrained, weights_path, device)

        self.model = models.Tripling(
            embed_dim=50,
            sgate_l1_dim=128,
            tgate_l1_dim=128,
            T=3,
            hidden_dims=[50, 50, 50],
            w_scale=0.01
        ).to(self.device)

        if pretrained or weights_path:
            self._load_weights(weights_path)

        self.model.eval()

    def _load_weights(self, weights_path: Optional[str] = None):
        weights_path = self._get_weights_path(weights_path)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"权重文件未找到: {weights_path}\n"
                "请下载预训练权重或设置 pretrained=False"
            )
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))

    def _setup_graph_input(self, state: torch.Tensor) -> Data:
        node_embed = self._get_init_node_embed(num_epochs=0)
        x = torch.cat((node_embed, state.detach().clone().unsqueeze(dim=1)), dim=-1)

        edges = list(self.graph.edges.keys())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor([self.graph.edges[e] for e in edges], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

    def _get_init_node_embed(self, num_epochs: int = 0) -> torch.Tensor:
        if self._node_embed is None:
            self._node_embed = models.get_init_node_embed(self.graph, num_epochs, self.device)
        return self._node_embed

    @torch.no_grad()
    def run(self, k: int, use_topk: bool = True) -> Set[int]:
        """执行算法选择种子节点。

        Args:
            k: 需要选择的种子节点数量。
            use_topk: 是否使用 topk 一次性选择，默认为 True。
                True: 一次性计算所有节点的 Q 值，选择 top-k。
                False: 迭代选择，每次选择一个节点后更新状态。

        Returns:
            Set[int]: 选中的种子节点集合。
        """
        state = torch.zeros(self.graph.num_nodes, dtype=torch.long)

        if use_topk:
            graph_input = self._setup_graph_input(state)
            loader = DataLoader([graph_input], batch_size=1, shuffle=False)
            for batch in loader:
                batch = batch.to(self.device)
                q_values = self.model(batch).squeeze()
            _, indices = torch.topk(q_values, k)
            return set(indices.tolist())

        selected = []
        for _ in range(k):
            graph_input = self._setup_graph_input(state)
            loader = DataLoader([graph_input], batch_size=1, shuffle=False)
            for batch in loader:
                batch = batch.to(self.device)
                q_values = self.model(batch).squeeze()

            q_values[state == 1] = -1e10
            action = torch.argmax(q_values).item()
            selected.append(action)
            state[action] = 1

        return set(selected)


class S2VDQNAlgorithm(BaseDLAlgorithm):
    """S2V-DQN 深度学习影响力最大化算法。

    使用 Structure2Vec 图神经网络结合 DQN 强化学习选择种子节点。

    References:
        Dai H, Khalil E B, Zhang Y, Dilkina B, Song L. Learning Combinatorial Optimization
        Algorithms over Graphs. Advances in Neural Information Processing Systems (NeurIPS), 2017.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import S2VDQNAlgorithm
        >>>
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = S2VDQNAlgorithm(graph, pretrained=True)
        >>> seeds = algo.run(k=10)
    """

    _weights_filename = 's2vdqn.ckpt'

    def __init__(
        self,
        graph: 'IMGraph',
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        device: str = 'auto'
    ):
        """初始化 S2V-DQN 算法。

        Args:
            graph: 输入图对象。
            pretrained: 是否使用预训练权重，默认为 True。
            weights_path: 本地权重路径，优先级高于 pretrained。
            device: 计算设备，支持 'auto'、'cpu'、'cuda'，默认为 'auto'。
        """
        super().__init__(graph, pretrained, weights_path, device)

        self.node_dim = 2
        self.edge_dim = 4

        self.model = models.S2V_DQN(
            reg_hidden=32,
            embed_dim=64,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            T=3,
            w_scale=0.01,
            avg=False
        ).to(self.device)

        if pretrained or weights_path:
            self._load_weights(weights_path)

        self.model.eval()

    def _load_weights(self, weights_path: Optional[str] = None):
        weights_path = self._get_weights_path(weights_path)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"权重文件未找到: {weights_path}\n"
                "请下载预训练权重或设置 pretrained=False"
            )
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))

    def _setup_graph_input(self, state: torch.Tensor) -> Data:
        x = torch.ones(self.graph.num_nodes, self.node_dim)
        x[:, 1] = 1 - state

        edges = list(self.graph.edges.keys())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        edge_attr = torch.ones(self.graph.num_edges, self.edge_dim)
        edge_weights = torch.tensor([self.graph.edges[e] for e in edges], dtype=torch.float)
        edge_attr[:, 1] = edge_weights
        edge_attr[:, 0] = state[edge_index[0]]
        edge_attr[:, 2] = torch.abs(state[edge_index[0]] - state[edge_index[1]])

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @torch.no_grad()
    def run(self, k: int, use_topk: bool = False) -> Set[int]:
        """执行算法选择种子节点。

        Args:
            k: 需要选择的种子节点数量。
            use_topk: 是否使用 topk 一次性选择，默认为 False。
                True: 一次性计算所有节点的 Q 值，选择 top-k。
                False: 迭代选择，每次选择一个节点后更新状态。

        Returns:
            Set[int]: 选中的种子节点集合。
        """
        state = torch.zeros(self.graph.num_nodes, dtype=torch.long)

        if use_topk:
            graph_input = self._setup_graph_input(state)
            loader = DataLoader([graph_input], batch_size=1, shuffle=False)
            for batch in loader:
                batch = batch.to(self.device)
                q_values = self.model(batch).squeeze()
            _, indices = torch.topk(q_values, k)
            return set(indices.tolist())

        selected = []
        for _ in range(k):
            graph_input = self._setup_graph_input(state)
            loader = DataLoader([graph_input], batch_size=1, shuffle=False)
            for batch in loader:
                batch = batch.to(self.device)
                q_values = self.model(batch).squeeze()

            q_values[state == 1] = -1e10
            action = torch.argmax(q_values).item()
            selected.append(action)
            state[action] = 1

        return set(selected)
