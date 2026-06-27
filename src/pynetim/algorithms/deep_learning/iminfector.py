"""IMInfector 影响力最大化算法。

基于嵌入的影响力最大化算法，利用 Inf2Vec 学习的 Source/Target 嵌入
计算影响概率矩阵，并通过贪心策略选择种子节点。
"""

from typing import Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...graph import IMGraph
    from ...embedding.inf2vec import Inf2Vec

from ..base_algorithm import BaseAlgorithm


class IMInfectorAlgorithm(BaseAlgorithm):
    """IMInfector 影响力最大化算法。

    基于学习到的 Source/Target 嵌入表示，计算影响概率矩阵 D = softmax(S · T^T)，
    使用贪心策略选择种子节点。每个候选种子的预算（影响节点数）由其 Source 嵌入
    的 L2 范数决定。

    使用方式:
        1. 先用 Inf2Vec 训练嵌入
        2. 将训练好的 Inf2Vec 模型传入本算法
        3. 调用 run(k) 选择种子

    References:
        Panagopoulos, G., Malliaros, F., & Vazirgiannis, M. (2020).
        Multi-task Learning for Influence Estimation and Maximization.
        IEEE Transactions on Knowledge and Data Engineering.

        Panagopoulos, G., Malliaros, F. D., & Vazirgiannis, M. (2020).
        Influence Maximization Using Influence and Susceptibility Embeddings.
        Proceedings of the International AAAI Conference on Web and Social Media,
        14, 511-521.

    Attributes:
        graph: 输入图对象。
        source_embeddings: 源嵌入矩阵。
        target_embeddings: 目标嵌入矩阵。

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.embedding import Inf2Vec
        >>> from pynetim.algorithms import IMInfectorAlgorithm
        >>>
        >>> graph = IMGraph(edges, weights=0.3)
        >>> cascades = [[0, 1, 3], [2, 4, 5, 1]]
        >>> inf2vec = Inf2Vec(graph, cascades=cascades, dimensions=64)
        >>> inf2vec.train()
        >>>
        >>> algo = IMInfectorAlgorithm(graph, inf2vec_model=inf2vec)
        >>> seeds = algo.run(k=5)
    """

    def __init__(
        self,
        graph: 'IMGraph',
        inf2vec_model: Optional['Inf2Vec'] = None,
        source_embeddings: Optional[np.ndarray] = None,
        target_embeddings: Optional[np.ndarray] = None,
        diffusion_model: Optional[str] = None,
    ):
        """初始化 IMInfector 算法。

        可以通过 Inf2Vec 模型或直接提供嵌入矩阵来初始化。

        Args:
            graph: IMGraph 图对象。
            inf2vec_model: 已训练的 Inf2Vec 模型。若提供，则从中提取嵌入。
            source_embeddings: 源嵌入矩阵，形状 (num_nodes, dim)。
                               若提供 inf2vec_model 则忽略此参数。
            target_embeddings: 目标嵌入矩阵，形状 (num_nodes, dim)。
                               若提供 inf2vec_model 则忽略此参数。
            diffusion_model: 扩散模型名称，支持 'IC' 或 'LT'，默认 None。

        Raises:
            ValueError: 未提供嵌入数据。
        """
        super().__init__(graph, diffusion_model)

        if inf2vec_model is not None:
            if inf2vec_model.source_embeddings is None:
                raise ValueError("Inf2Vec 模型尚未训练，请先调用 train() 方法。")
            self.source_embeddings = inf2vec_model.source_embeddings
            self.target_embeddings = inf2vec_model.target_embeddings
        elif source_embeddings is not None and target_embeddings is not None:
            self.source_embeddings = source_embeddings
            self.target_embeddings = target_embeddings
        else:
            raise ValueError(
                "必须提供 inf2vec_model 或 source_embeddings/target_embeddings。"
            )

        self._influence_matrix: Optional[np.ndarray] = None

    def _compute_influence_matrix(self) -> np.ndarray:
        """计算影响概率矩阵 D = softmax(S · T^T)。

        Returns:
            np.ndarray: 影响概率矩阵，形状为 (num_nodes, num_nodes)。
        """
        D = np.dot(self.source_embeddings, self.target_embeddings.T)

        D = D - np.max(D, axis=1, keepdims=True)
        exp_D = np.exp(D)
        D = exp_D / np.sum(exp_D, axis=1, keepdims=True)

        return D

    def _compute_budgets(self, top_p: float = 100.0) -> tuple:
        """计算每个候选节点的预算（影响节点数）。

        基于 Source 嵌入的 L2 范数，按比例分配预算。

        Args:
            top_p: 选择 Source 范数前 top_p% 的节点作为候选种子。

        Returns:
            tuple: (chosen_indices, budgets) 候选节点索引和对应预算。
        """
        norms = np.sum(self.source_embeddings ** 2, axis=1)

        num_nodes = self.graph.num_nodes
        num_candidates = max(1, int(top_p * num_nodes / 100.0))
        chosen = np.argsort(-norms)[:num_candidates]

        chosen_norms = norms[chosen]
        total_norm = np.sum(chosen_norms)

        if total_norm == 0:
            budgets = np.ones(num_candidates, dtype=np.float64)
        else:
            budgets = num_nodes * chosen_norms / total_norm

        budgets = np.rint(budgets).astype(int)
        budgets = np.maximum(budgets, 1)

        return chosen, budgets

    def run(self, k: int, top_p: float = 100.0) -> Set[int]:
        """执行 IMInfector 算法选择种子节点。

        基于影响概率矩阵 D，使用类似 CELF 的贪心策略选择种子。
        每个种子的预算（影响节点数）由其 Source 嵌入的 L2 范数决定。

        Args:
            k: 种子数量。
            top_p: 选择 Source 范数前 top_p% 的节点作为候选种子。默认 100.0。

        Returns:
            Set[int]: 选出的种子节点集合。
        """
        D = self._compute_influence_matrix()
        chosen, budgets = self._compute_budgets(top_p)

        num_targets = D.shape[1]
        influenced = np.zeros(num_targets)
        total_set = set(range(num_targets))
        seeds = []

        Q = []
        for i, cand in enumerate(chosen):
            budget = int(budgets[i])
            uninfected = list(total_set - set(np.where(influenced)[0]))
            if len(uninfected) > 0 and budget <= len(uninfected):
                top_indices = np.argpartition(D[cand, uninfected], -budget)[-budget:]
                spread = sum(D[cand, uninfected][top_indices])
            elif len(uninfected) > 0:
                spread = sum(D[cand, uninfected])
            else:
                spread = 0.0
            Q.append([int(cand), spread, 0])

        while len(seeds) < k and len(Q) > 0:
            u = Q[0]
            cand = u[0]

            if u[2] == len(seeds):
                budget = int(budgets[np.where(chosen == cand)[0][0]])
                uninfected = list(total_set - set(np.where(influenced)[0]))
                if len(uninfected) > 0 and budget > 0:
                    actual_budget = min(budget, len(uninfected))
                    top_indices = np.argpartition(D[cand, uninfected], -actual_budget)[-actual_budget:]
                    influenced_nodes = np.array(uninfected)[top_indices]
                    influenced[influenced_nodes] = 1

                seeds.append(cand)
                Q = [item for item in Q if item[0] != cand]
            else:
                budget = int(budgets[np.where(chosen == cand)[0][0]])
                uninfected = list(total_set - set(np.where(influenced)[0]))
                if len(uninfected) > 0 and budget > 0:
                    actual_budget = min(budget, len(uninfected))
                    top_indices = np.argpartition(D[cand, uninfected], -actual_budget)[-actual_budget:]
                    spread = sum(D[cand, uninfected][top_indices])
                else:
                    spread = 0.0

                u[1] = spread
                u[2] = len(seeds)
                Q = sorted(Q, key=lambda x: x[1], reverse=True)

        self.seeds = seeds
        return set(seeds)

    def get_influence_matrix(self) -> np.ndarray:
        """获取影响概率矩阵。

        Returns:
            np.ndarray: 影响概率矩阵，形状为 (num_nodes, num_nodes)。
        """
        if self._influence_matrix is None:
            self._influence_matrix = self._compute_influence_matrix()
        return self._influence_matrix

    def get_influence_probability(self, source: int, target: int) -> float:
        """计算节点 source 对节点 target 的影响概率。

        Args:
            source: 源节点 ID。
            target: 目标节点 ID。

        Returns:
            float: 影响概率。
        """
        D = self.get_influence_matrix()
        return float(D[source, target])

    def get_influence_weights(self) -> Dict[tuple, float]:
        """计算所有边的影响权重（归一化）。

        Returns:
            Dict[tuple, float]: 边到影响权重的映射。
        """
        D = self.get_influence_matrix()
        result: Dict[tuple, float] = {}

        for u in range(self.graph.num_nodes):
            for v in self.graph.out_neighbors(u):
                result[(u, v)] = float(D[u, v])

        return result
