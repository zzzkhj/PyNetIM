import os
import random
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ..base_drl import BaseDRLAlgorithm
from . import models
from .....weights import WeightManager

if TYPE_CHECKING:
    from .....graph import IMGraph

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)


class ToupleGDDAlgorithm(BaseDRLAlgorithm):
    """ToupleGDD 深度学习影响力最大化算法。

    使用三重门控图神经网络 (Tripling GNN) 结合强化学习选择种子节点。

    References:
        Chen, T., Yan, S., Guo, J., & Wu, W. (2024). ToupleGDD: A Fine-Designed 
        Solution of Influence Maximization by Deep Reinforcement Learning. 
        IEEE Transactions on Computational Social Systems.

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
        device: str = 'auto',
        use_deepwalk: bool = False,
        deepwalk_epochs: int = 0,
        **kwargs
    ):
        """初始化 ToupleGDD 算法。

        Args:
            graph: 输入图对象。
            pretrained: 是否使用预训练权重，默认为 True。
            weights_path: 本地权重路径，优先级高于 pretrained。
            device: 计算设备，支持 'auto'、'cpu'、'cuda'，默认为 'auto'。
            use_deepwalk: 是否使用 DeepWalk 学习节点嵌入，默认为 False。
                - True: 使用 DeepWalk 模型学习节点嵌入（较慢）。
                - False: 使用随机初始化的嵌入向量（较快）。
            deepwalk_epochs: DeepWalk 训练轮次，默认为 0。
                仅当 use_deepwalk=True 时有效。
            **kwargs: 传递给父类的其他参数，包括：
                - diffusion_model: 扩散模型名称，支持 'IC' 或 'LT'
        """
        super().__init__(graph, pretrained, weights_path, device, **kwargs)
        
        self.use_deepwalk = use_deepwalk
        self.deepwalk_epochs = deepwalk_epochs
        self.embed_dim = 50

        self.model = models.Tripling(
            embed_dim=self.embed_dim,
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
        if weights_path is not None:
            weights_file = weights_path
        else:
            weights_file = WeightManager.get_weights_path(
                self._weights_filename, 
                verbose=False
            )
        
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"权重文件未找到: {weights_file}\n"
                "请下载预训练权重或设置 pretrained=False"
            )
        self.model.load_state_dict(torch.load(weights_file, map_location=self.device, weights_only=True))

    def _setup_graph_input(self, state: torch.Tensor) -> Data:
        node_embed = self._get_init_node_embed()
        x = torch.cat((node_embed, state.detach().clone().unsqueeze(dim=1)), dim=-1)

        edges = list(self.graph.edges.keys())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor([self.graph.edges[e] for e in edges], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

    def _get_init_node_embed(self) -> torch.Tensor:
        if self._node_embed is None:
            if self.use_deepwalk:
                self._node_embed = models.get_init_node_embed(self.graph, self.deepwalk_epochs, self.device)
            else:
                self._node_embed = torch.randn(self.graph.num_nodes, self.embed_dim * 2, device=self.device)
        return self._node_embed

    def _prepare_inference(self):
        """准备推理环境。"""
        self.model.eval()

    def _init_state(self) -> torch.Tensor:
        """初始化状态。

        Returns:
            torch.Tensor: 初始状态向量（全 0）。
        """
        return torch.zeros(self.graph.num_nodes, dtype=torch.long)

    def _compute_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """计算 Q 值。

        Args:
            state: 当前状态向量。

        Returns:
            torch.Tensor: 各节点的 Q 值。
        """
        graph_input = self._setup_graph_input(state)
        loader = DataLoader([graph_input], batch_size=1, shuffle=False)
        for batch in loader:
            batch = batch.to(self.device)
            return self.model(batch).squeeze()

    def _update_state(self, state: torch.Tensor, node: int) -> torch.Tensor:
        """更新状态。

        Args:
            state: 当前状态向量。
            node: 新选择的节点。

        Returns:
            torch.Tensor: 更新后的状态向量。
        """
        state[node] = 1
        return state

    def _mask_selected(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """屏蔽已选节点的 Q 值。

        Args:
            q_values: Q 值张量。
            state: 当前状态向量。

        Returns:
            torch.Tensor: 屏蔽后的 Q 值。
        """
        q_values[state == 1] = -1e10
        return q_values


class S2VDQNAlgorithm(BaseDRLAlgorithm):
    """S2V-DQN 深度学习影响力最大化算法。

    使用 Structure2Vec 图神经网络结合 DQN 强化学习选择种子节点。

    References:
        Dai, H., Khalil, E. B., Zhang, Y., Dilkina, B., & Song, L. (2017). 
        Learning Combinatorial Optimization Algorithms over Graphs. 
        Advances in Neural Information Processing Systems (NeurIPS).

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
        if weights_path is not None:
            weights_file = weights_path
        else:
            weights_file = WeightManager.get_weights_path(
                self._weights_filename, 
                verbose=False
            )
        
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"权重文件未找到: {weights_file}\n"
                "请下载预训练权重或设置 pretrained=False"
            )
        self.model.load_state_dict(torch.load(weights_file, map_location=self.device, weights_only=True))

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

    def _prepare_inference(self):
        """准备推理环境。"""
        self.model.eval()

    def _init_state(self) -> torch.Tensor:
        """初始化状态。

        Returns:
            torch.Tensor: 初始状态向量（全 0）。
        """
        return torch.zeros(self.graph.num_nodes, dtype=torch.long)

    def _compute_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """计算 Q 值。

        Args:
            state: 当前状态向量。

        Returns:
            torch.Tensor: 各节点的 Q 值。
        """
        graph_input = self._setup_graph_input(state)
        loader = DataLoader([graph_input], batch_size=1, shuffle=False)
        for batch in loader:
            batch = batch.to(self.device)
            return self.model(batch).squeeze()

    def _update_state(self, state: torch.Tensor, node: int) -> torch.Tensor:
        """更新状态。

        Args:
            state: 当前状态向量。
            node: 新选择的节点。

        Returns:
            torch.Tensor: 更新后的状态向量。
        """
        state[node] = 1
        return state

    def _mask_selected(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """屏蔽已选节点的 Q 值。

        Args:
            q_values: Q 值张量。
            state: 当前状态向量。

        Returns:
            torch.Tensor: 屏蔽后的 Q 值。
        """
        q_values[state == 1] = -1e10
        return q_values
