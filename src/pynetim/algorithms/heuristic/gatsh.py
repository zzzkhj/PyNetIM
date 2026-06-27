"""GATSH: Graph Attention Network with Structural Holes for Influence Maximization.

Based on: "Social Network Influence Maximization Based on Graph Attention Mechanisms"
ICETIS 2024

Note: This algorithm uses GAT's attention weights without training,
so it is classified as a heuristic algorithm rather than deep learning.
"""

from __future__ import annotations

from typing import Set, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected

from ..base_algorithm import BaseAlgorithm

if TYPE_CHECKING:
    from ...graph import IMGraph


class GATSHModel(nn.Module):
    """GATSH 模型。

    结合图注意力网络和结构洞理论的影响力最大化模型。

    Args:
        in_dim: 输入特征维度。
        hidden_dim: 隐藏层维度，默认 64。
        out_dim: 输出特征维度，默认 64。
        heads: 注意力头数，默认 8。
        dropout: Dropout 概率，默认 0.6。

    Example:
        >>> model = GATSHModel(in_dim=64, hidden_dim=64, out_dim=64)
        >>> node_scores, attention = model(x, edge_index)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        heads: int = 8,
        dropout: float = 0.6
    ):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        """前向传播。

        Args:
            x: 节点特征矩阵。
            edge_index: 边索引。

        Returns:
            tuple: (节点影响力分数, 注意力权重)。
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x, attention = self.conv2(x, edge_index, return_attention_weights=True)

        node_scores = torch.norm(x, p=2, dim=-1)

        return node_scores, attention


class GATSHAlgorithm(BaseAlgorithm):
    """基于图注意力网络和结构洞的影响力最大化算法。

    该算法结合图注意力网络(GAT)和结构洞(Structural Holes)理论来评估节点影响力。
    使用GAT学习节点的特征表示，然后基于注意力权重计算节点的约束系数，
    约束系数越小，节点越容易成为结构洞节点，影响力越大。

    Note: 此算法使用GAT的注意力权重但不进行训练，因此归类为启发式算法。

    References:
        Social Network Influence Maximization Based on Graph Attention Mechanisms.
        ICETIS 2024.

    Attributes:
        hidden_dim: 隐藏层维度。
        out_dim: 输出特征维度。
        heads: 注意力头数。
        dropout: Dropout 概率。
        device: 计算设备。

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import GATSHAlgorithm
        >>>
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = GATSHAlgorithm(graph, hidden_dim=64, heads=8)
        >>> seeds = algo.run(k=10)
    """

    def __init__(
        self,
        graph: 'IMGraph',
        hidden_dim: int = 64,
        out_dim: int = 64,
        heads: int = 8,
        dropout: float = 0.6,
        device: str = 'auto',
        diffusion_model: str = None
    ):
        """初始化 GATSH 算法。

        Args:
            graph: 输入图对象。
            hidden_dim: 隐藏层维度，默认 64。
            out_dim: 输出特征维度，默认 64。
            heads: 注意力头数，默认 8。
            dropout: Dropout 概率，默认 0.6。
            device: 计算设备，支持 'auto'、'cpu'、'cuda'，默认为 'auto'。
            diffusion_model: 扩散模型名称，支持 'IC' 或 'LT'，默认为 None。
        """
        super().__init__(graph, diffusion_model)

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self._build_model()

    def _build_model(self):
        """构建模型。"""
        in_dim = self._get_node_features().shape[1]

        self.model = GATSHModel(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            heads=self.heads,
            dropout=self.dropout
        ).to(self.device)

    def _get_node_features(self) -> torch.Tensor:
        """获取节点特征。

        使用度、邻居平均度等结构特征作为初始节点特征。

        Returns:
            torch.Tensor: 节点特征矩阵。
        """
        num_nodes = self.graph.num_nodes

        degrees = torch.tensor(
            [self.graph.out_degree(i) for i in range(num_nodes)],
            dtype=torch.float32
        )

        in_degrees = torch.tensor(
            [self.graph.in_degree(i) for i in range(num_nodes)],
            dtype=torch.float32
        )

        features = torch.stack([
            degrees,
            in_degrees,
            degrees / (degrees.max() + 1e-8),
            in_degrees / (in_degrees.max() + 1e-8),
            torch.log1p(degrees),
            torch.log1p(in_degrees),
            torch.ones(num_nodes),
            torch.arange(num_nodes, dtype=torch.float32) / num_nodes,
        ], dim=1)

        features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)

        return features

    def _get_edge_index(self) -> torch.Tensor:
        """获取边索引。

        Returns:
            torch.Tensor: 边索引，形状为 [2, num_edges]。
        """
        edges = []
        for u in range(self.graph.num_nodes):
            for v in self.graph.out_neighbors(u):
                edges.append([u, v])

        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        edge_index = torch.tensor(edges, dtype=torch.long).t()

        edge_index = to_undirected(edge_index, num_nodes=self.graph.num_nodes)

        return edge_index.to(self.device)

    def _compute_constraint_coefficients(
        self,
        attention_weights: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """计算节点的约束系数。

        约束系数越小，节点越容易成为结构洞节点，影响力越大。

        公式: GATSH(i) = Σ_{j∈F(i)} (a_ij * Σ_{q∈Φ(i,j)} a_iq * a_qj)

        Args:
            attention_weights: 注意力权重，形状为 [num_edges, 1]。
            edge_index: 边索引，形状为 [2, num_edges]。

        Returns:
            torch.Tensor: 每个节点的约束系数。
        """
        num_nodes = self.graph.num_nodes
        attention_weights = attention_weights.squeeze()
        edge_index = edge_index.long()

        adj_attention = torch.zeros((num_nodes, num_nodes), device=self.device)
        adj_attention[edge_index[0], edge_index[1]] = attention_weights

        row_sum = adj_attention.sum(dim=1, keepdim=True)
        adj_attention_norm = adj_attention / (row_sum + 1e-8)

        constraint = torch.zeros(num_nodes, device=self.device)

        for i in range(num_nodes):
            neighbors_i = (adj_attention_norm[i] > 0).nonzero(as_tuple=True)[0]

            if len(neighbors_i) == 0:
                constraint[i] = 1.0
                continue

            for j in neighbors_i:
                a_ij = adj_attention_norm[i, j]

                neighbors_j = (adj_attention_norm[j] > 0).nonzero(as_tuple=True)[0]
                common_neighbors = set(neighbors_i.tolist()) & set(neighbors_j.tolist())

                inner_sum = 0.0
                for q in common_neighbors:
                    if q != i and q != j:
                        a_iq = adj_attention_norm[i, q]
                        a_qj = adj_attention_norm[q, j]
                        inner_sum += (a_iq * a_qj).item()

                constraint[i] += a_ij.item() * (a_ij.item() + inner_sum)

        return constraint

    @torch.no_grad()
    def run(self, k: int) -> Set[int]:
        """执行算法选择种子节点。

        Args:
            k: 种子节点数量。

        Returns:
            set: 选择的种子节点集合。
        """
        self.model.eval()

        x = self._get_node_features().to(self.device)
        edge_index = self._get_edge_index()

        node_scores, attention = self.model(x, edge_index)

        attention_edge_index, attention_weights = attention

        if attention_weights.dim() > 1:
            attention_weights = attention_weights.mean(dim=-1)

        constraint = self._compute_constraint_coefficients(
            attention_weights.cpu(),
            attention_edge_index.cpu()
        )

        _, indices = torch.topk(constraint, k, largest=False)

        seed_list = indices.tolist()
        self.seeds = seed_list
        return set(seed_list)
