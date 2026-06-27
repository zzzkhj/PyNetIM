"""Node2Vec 图嵌入算法。"""

import random
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pynetim.graph import IMGraph

from .base import BaseEmbedding


class Node2Vec(BaseEmbedding):
    """Node2Vec 图节点嵌入算法。

    通过有偏随机游走学习节点的低维表示。结合了 BFS 和 DFS 的特点，
    可以捕获节点的同质性（homophily）和结构等价性（structural equivalence）。

    时间复杂度: O(num_walks * walk_length * num_nodes)

    Attributes:
        graph: 输入图对象。
        dimensions: 嵌入维度。
        p: 返回参数。
        q: 进出参数。
        embeddings: 学习到的节点嵌入矩阵。

    References:
        Grover, A., & Leskovec, J. (2016). node2vec: Scalable Feature Learning
        for Networks. KDD, 855-864.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.embedding import Node2Vec
        >>>
        >>> graph = IMGraph(edges, weights=0.3)
        >>> model = Node2Vec(graph, dimensions=64, p=1, q=1)
        >>> model.train()
        >>> embeddings = model.get_all_embeddings()
    """

    def __init__(
        self,
        graph: 'IMGraph',
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        """初始化 Node2Vec 模型。

        Args:
            graph: IMGraph 图对象。
            dimensions: 嵌入维度，默认 128。
            walk_length: 每次随机游走的长度，默认 80。
            num_walks: 每个节点的游走次数，默认 10。
            p: 返回参数，控制回访上一个节点的概率。p > 1 降低回访概率，
               p < 1 增加回访概率。默认 1.0。
            q: 进出参数，控制游走方向。q > 1 倾向于 BFS（局部），
               q < 1 倾向于 DFS（全局）。默认 1.0。
            random_seed: 随机种子，默认 None（使用全局种子）。
        """
        super().__init__(graph, dimensions, walk_length, num_walks, random_seed)

        self.p = p
        self.q = q
        self._alias_nodes = {}
        self._alias_edges = {}

    def _preprocess_transition_probs(self) -> None:
        """预处理转移概率，使用别名采样法加速随机游走。"""
        self._alias_nodes = {}
        self._alias_edges = {}

        for node in range(self.graph.num_nodes):
            neighbors = list(self.graph.out_neighbors(node))
            if len(neighbors) == 0:
                continue

            weights = []
            for neighbor in neighbors:
                w = self.graph.get_edge_weight(node, neighbor) if self.graph.has_edge(node, neighbor) else 1.0
                weights.append(w)

            norm_weights = np.array(weights) / sum(weights)
            self._alias_nodes[node] = self._create_alias_table(norm_weights, neighbors)

        for node in range(self.graph.num_nodes):
            neighbors = list(self.graph.out_neighbors(node))
            for neighbor in neighbors:
                prev_node = node
                curr_node = neighbor
                self._alias_edges[(prev_node, curr_node)] = self._get_alias_edge(prev_node, curr_node)

    def _get_alias_edge(self, prev_node: int, curr_node: int) -> tuple:
        """计算边的别名采样表。

        Args:
            prev_node: 前一个节点。
            curr_node: 当前节点。

        Returns:
            tuple: (items, alias, prob)
        """
        neighbors = list(self.graph.out_neighbors(curr_node))
        if len(neighbors) == 0:
            return ([], [], [])

        weights = []
        for neighbor in neighbors:
            w = self.graph.get_edge_weight(curr_node, neighbor) if self.graph.has_edge(curr_node, neighbor) else 1.0

            if neighbor == prev_node:
                w = w / self.p
            elif self.graph.has_edge(neighbor, prev_node) or self.graph.has_edge(prev_node, neighbor):
                w = w
            else:
                w = w / self.q

            weights.append(w)

        norm_weights = np.array(weights) / sum(weights)
        return self._create_alias_table(norm_weights, neighbors)

    def _create_alias_table(self, probs: np.ndarray, items: List) -> tuple:
        """创建别名采样表。

        Args:
            probs: 概率分布。
            items: 对应的元素列表。

        Returns:
            tuple: (items, alias, prob)
        """
        n = len(probs)
        if n == 0:
            return (items, [], [])

        alias = np.zeros(n, dtype=np.int32)
        prob = np.zeros(n, dtype=np.float64)

        scaled_probs = probs * n
        small = []
        large = []

        for i, p in enumerate(scaled_probs):
            if p < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            small_idx = small.pop()
            large_idx = large.pop()

            prob[small_idx] = scaled_probs[small_idx]
            alias[small_idx] = large_idx

            scaled_probs[large_idx] = scaled_probs[large_idx] + scaled_probs[small_idx] - 1.0

            if scaled_probs[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        while large:
            prob[large.pop()] = 1.0

        while small:
            prob[small.pop()] = 1.0

        return (items, alias, prob)

    def _alias_sample(self, alias_table: tuple) -> int:
        """使用别名采样法进行采样。

        Args:
            alias_table: 别名采样表。

        Returns:
            int: 采样结果。
        """
        items, alias, prob = alias_table
        n = len(items)

        if n == 0:
            return items[0] if items else -1

        i = random.randint(0, n - 1)
        if random.random() < prob[i]:
            return items[i]
        else:
            return items[alias[i]]

    def _do_walk(self, start_node: int) -> List[int]:
        """从指定节点开始执行一次有偏随机游走。

        Args:
            start_node: 起始节点。

        Returns:
            List[int]: 游走路径。
        """
        walk = [start_node]

        while len(walk) < self.walk_length:
            curr_node = walk[-1]
            neighbors = list(self.graph.out_neighbors(curr_node))

            if len(neighbors) == 0:
                break

            if len(walk) == 1:
                if curr_node not in self._alias_nodes:
                    break
                next_node = self._alias_sample(self._alias_nodes[curr_node])
            else:
                prev_node = walk[-2]
                edge_key = (prev_node, curr_node)
                if edge_key not in self._alias_edges:
                    if curr_node not in self._alias_nodes:
                        break
                    next_node = self._alias_sample(self._alias_nodes[curr_node])
                else:
                    next_node = self._alias_sample(self._alias_edges[edge_key])

            if next_node == -1:
                break

            walk.append(next_node)

        return walk

    def train(
        self,
        window_size: int = 10,
        min_count: int = 1,
        epochs: int = 5,
        learning_rate: float = 0.025,
        negative_samples: int = 5,
        backend: str = 'auto',
    ) -> None:
        """训练 Node2Vec 模型。

        Args:
            window_size: 上下文窗口大小，默认 10。
            min_count: 忽略出现次数少于此值的节点，默认 1。
            epochs: 训练轮数，默认 5。
            learning_rate: 初始学习率，默认 0.025。
            negative_samples: 负采样数量，默认 5。
            backend: 后端实现，支持 'auto'、'gensim'、'pytorch'。
                     'auto' 优先使用 gensim，不可用时使用 pytorch。默认 'auto'。
        """
        self._preprocess_transition_probs()
        self._generate_walks()
        self._train_skipgram(window_size, min_count, epochs, learning_rate, negative_samples, backend)
