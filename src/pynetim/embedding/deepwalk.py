"""DeepWalk 图嵌入算法。"""

import random
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pynetim.graph import IMGraph

from .base import BaseEmbedding


class DeepWalk(BaseEmbedding):
    """DeepWalk 图节点嵌入算法。

    通过均匀随机游走学习节点的低维表示。将节点视为"单词"，
    游走序列视为"句子"，使用 Skip-gram 模型学习节点嵌入。

    DeepWalk 是 Node2Vec 的特例（p=1, q=1），不使用有偏采样。

    时间复杂度: O(num_walks * walk_length * num_nodes)

    Attributes:
        graph: 输入图对象。
        dimensions: 嵌入维度。
        embeddings: 学习到的节点嵌入矩阵。

    References:
        Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). DeepWalk: Online Learning
        of Social Representations. KDD, 701-710.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.embedding import DeepWalk
        >>>
        >>> graph = IMGraph(edges, weights=0.3)
        >>> model = DeepWalk(graph, dimensions=64)
        >>> model.train()
        >>> embeddings = model.get_all_embeddings()
    """

    def __init__(
        self,
        graph: 'IMGraph',
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        random_seed: Optional[int] = None,
    ):
        """初始化 DeepWalk 模型。

        Args:
            graph: IMGraph 图对象。
            dimensions: 嵌入维度，默认 128。
            walk_length: 每次随机游走的长度，默认 80。
            num_walks: 每个节点的游走次数，默认 10。
            random_seed: 随机种子，默认 None（使用全局种子）。
        """
        super().__init__(graph, dimensions, walk_length, num_walks, random_seed)

    def _do_walk(self, start_node: int) -> List[int]:
        """从指定节点开始执行一次均匀随机游走。

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

            next_node = random.choice(neighbors)
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
        """训练 DeepWalk 模型。

        Args:
            window_size: 上下文窗口大小，默认 10。
            min_count: 忽略出现次数少于此值的节点，默认 1。
            epochs: 训练轮数，默认 5。
            learning_rate: 初始学习率，默认 0.025。
            negative_samples: 负采样数量，默认 5。
            backend: 后端实现，支持 'auto'、'gensim'、'pytorch'。
                     'auto' 优先使用 gensim，不可用时使用 pytorch。默认 'auto'。
        """
        self._generate_walks()
        self._train_skipgram(window_size, min_count, epochs, learning_rate, negative_samples, backend)
