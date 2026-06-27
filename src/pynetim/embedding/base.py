"""图嵌入算法基类。"""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pynetim.graph import IMGraph


def _get_effective_seed(local_seed: Optional[int] = None) -> Optional[int]:
    """获取有效种子：局部 > 全局 > None。"""
    if local_seed is not None:
        return local_seed
    from pynetim.random import get_random_seed
    return get_random_seed()


class BaseEmbedding(ABC):
    """图嵌入算法基类。

    所有图嵌入算法的抽象基类，定义了通用接口。

    Attributes:
        graph: 输入图对象。
        dimensions: 嵌入维度。
        embeddings: 学习到的节点嵌入矩阵。

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.embedding import Node2Vec
        >>>
        >>> graph = IMGraph(edges, weights=0.3)
        >>> model = Node2Vec(graph, dimensions=64)
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
        """初始化嵌入模型。

        Args:
            graph: IMGraph 图对象。
            dimensions: 嵌入维度，默认 128。
            walk_length: 每次随机游走的长度，默认 80。
            num_walks: 每个节点的游走次数，默认 10。
            random_seed: 随机种子，默认 None（使用全局种子）。
        """
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.random_seed = _get_effective_seed(random_seed)
        self.embeddings: Optional[np.ndarray] = None
        self._walks: List[List[int]] = []

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    @abstractmethod
    def _do_walk(self, start_node: int) -> List[int]:
        """从指定节点开始执行一次随机游走。

        Args:
            start_node: 起始节点。

        Returns:
            List[int]: 游走路径。
        """
        pass

    @abstractmethod
    def train(
        self,
        window_size: int = 10,
        min_count: int = 1,
        epochs: int = 5,
        learning_rate: float = 0.025,
        negative_samples: int = 5,
        backend: str = 'auto',
    ) -> None:
        """训练嵌入模型。

        Args:
            window_size: 上下文窗口大小，默认 10。
            min_count: 忽略出现次数少于此值的节点，默认 1。
            epochs: 训练轮数，默认 5。
            learning_rate: 初始学习率，默认 0.025。
            negative_samples: 负采样数量，默认 5。
            backend: 后端实现，支持 'auto'、'gensim'、'pytorch'。
        """
        pass

    def _generate_walks(self) -> None:
        """生成所有随机游走序列。"""
        import random

        self._walks = []
        nodes = list(range(self.graph.num_nodes))

        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self._do_walk(node)
                if len(walk) > 1:
                    self._walks.append(walk)

    def _train_skipgram(
        self,
        window_size: int,
        min_count: int,
        epochs: int,
        learning_rate: float,
        negative_samples: int,
        backend: str,
    ) -> None:
        """使用 Skip-gram 模型训练嵌入。

        Args:
            window_size: 上下文窗口大小。
            min_count: 最小出现次数。
            epochs: 训练轮数。
            learning_rate: 学习率。
            negative_samples: 负采样数量。
            backend: 后端实现。
        """
        import random
        import warnings

        if backend == 'auto':
            try:
                import gensim
                backend = 'gensim'
            except ImportError:
                backend = 'pytorch'

        if backend == 'gensim':
            self._train_gensim(window_size, min_count, epochs, learning_rate, negative_samples)
        elif backend == 'pytorch':
            self._train_pytorch(window_size, epochs, learning_rate, negative_samples)
        else:
            raise ValueError(f"不支持的后端: {backend}，请选择 'gensim' 或 'pytorch'")

    def _train_gensim(
        self,
        window_size: int,
        min_count: int,
        epochs: int,
        learning_rate: float,
        negative_samples: int,
    ) -> None:
        """使用 gensim 的 Word2Vec 训练。"""
        try:
            from gensim.models import Word2Vec
        except ImportError:
            raise ImportError(
                "gensim 未安装。请使用 'pip install gensim' 安装，"
                "或使用 backend='pytorch' 训练。"
            )

        walks_str = [[str(node) for node in walk] for walk in self._walks]

        model = Word2Vec(
            sentences=walks_str,
            vector_size=self.dimensions,
            window=window_size,
            min_count=min_count,
            sg=1,
            workers=1,
            epochs=epochs,
            alpha=learning_rate,
            negative=negative_samples,
        )

        self.embeddings = np.zeros((self.graph.num_nodes, self.dimensions))
        for node in range(self.graph.num_nodes):
            node_str = str(node)
            if node_str in model.wv:
                self.embeddings[node] = model.wv[node_str]

    def _train_pytorch(
        self,
        window_size: int,
        epochs: int,
        learning_rate: float,
        negative_samples: int,
    ) -> None:
        """使用 PyTorch 实现 Skip-gram 训练。"""
        import random
        import warnings

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError(
                "PyTorch 未安装。请使用 'pip install torch' 安装，"
                "或使用 backend='gensim' 训练。"
            )

        vocab_size = self.graph.num_nodes

        class SkipGram(nn.Module):
            def __init__(self, vocab_size, embed_dim):
                super().__init__()
                self.in_embed = nn.Embedding(vocab_size, embed_dim)
                self.out_embed = nn.Embedding(vocab_size, embed_dim)

                nn.init.xavier_uniform_(self.in_embed.weight)
                nn.init.xavier_uniform_(self.out_embed.weight)

            def forward(self, center, context, neg_samples):
                center_embed = self.in_embed(center)
                context_embed = self.out_embed(context)
                neg_embed = self.out_embed(neg_samples)

                pos_score = torch.sum(center_embed * context_embed, dim=1)
                pos_score = torch.clamp(pos_score, -10, 10)
                pos_score = -torch.nn.functional.logsigmoid(pos_score)

                neg_score = torch.bmm(neg_embed, center_embed.unsqueeze(2)).squeeze(2)
                neg_score = torch.clamp(neg_score, -10, 10)
                neg_score = -torch.nn.functional.logsigmoid(-neg_score)
                neg_score = torch.sum(neg_score, dim=1)

                return torch.mean(pos_score + neg_score)

        training_data = []
        for walk in self._walks:
            for i, center in enumerate(walk):
                start = max(0, i - window_size)
                end = min(len(walk), i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        training_data.append((center, walk[j]))

        if len(training_data) == 0:
            warnings.warn("没有生成训练数据，请检查图结构。")
            self.embeddings = np.zeros((vocab_size, self.dimensions))
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SkipGram(vocab_size, self.dimensions).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        batch_size = 512
        for epoch in range(epochs):
            random.shuffle(training_data)
            total_loss = 0.0
            num_batches = 0

            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]

                centers = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
                contexts = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)

                neg_samples_tensor = torch.randint(
                    0, vocab_size,
                    (len(batch), negative_samples),
                    dtype=torch.long,
                    device=device
                )

                optimizer.zero_grad()
                loss = model(centers, contexts, neg_samples_tensor)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.embeddings = model.in_embed.weight.detach().cpu().numpy()

    def get_embedding(self, node: int) -> np.ndarray:
        """获取指定节点的嵌入向量。

        Args:
            node: 节点 ID。

        Returns:
            np.ndarray: 节点的嵌入向量。

        Raises:
            ValueError: 模型尚未训练。
        """
        if self.embeddings is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法。")
        return self.embeddings[node]

    def get_all_embeddings(self) -> np.ndarray:
        """获取所有节点的嵌入矩阵。

        Returns:
            np.ndarray: 嵌入矩阵，形状为 (num_nodes, dimensions)。

        Raises:
            ValueError: 模型尚未训练。
        """
        if self.embeddings is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法。")
        return self.embeddings

    def most_similar(self, node: int, topn: int = 10) -> List[tuple]:
        """找出与指定节点最相似的节点。

        Args:
            node: 目标节点 ID。
            topn: 返回的相似节点数量，默认 10。

        Returns:
            List[tuple]: 相似节点列表，每个元素为 (node_id, similarity)。

        Raises:
            ValueError: 模型尚未训练。
        """
        if self.embeddings is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法。")

        node_vec = self.embeddings[node]

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = self.embeddings / norms

        node_norm = node_vec / (np.linalg.norm(node_vec) + 1e-10)
        similarities = np.dot(normalized, node_norm)

        most_similar_indices = np.argsort(similarities)[::-1][1:topn + 1]

        return [(int(idx), float(similarities[idx])) for idx in most_similar_indices]
