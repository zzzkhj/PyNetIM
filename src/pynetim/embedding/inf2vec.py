"""Inf2Vec 社交影响嵌入算法。"""

import random
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pynetim.graph import IMGraph

from .base import BaseEmbedding, _get_effective_seed


class Inf2Vec(BaseEmbedding):
    """Inf2Vec 社交影响嵌入算法。

    基于扩散级联数据学习节点的影响嵌入表示。与 Node2Vec/DeepWalk 不同，
    Inf2Vec 不仅利用网络结构，还结合扩散级联信息来捕获节点间的影响关系。

    每个节点拥有两个嵌入向量:
        - Source embedding S_u: 节点 u 影响他人的能力
        - Target embedding T_u: 节点 u 被他人影响的趋势

    算法流程:
        1. 从扩散级联中提取影响传播网络
        2. 在影响传播网络上进行随机游走，生成影响上下文
        3. 结合全局用户相似性上下文（可选）
        4. 使用 Skip-gram（带 Source/Target 分离）训练节点嵌入

    References:
        Kang, C., Yin, D., Cheng, R., Agrawal, D., & Liao, X. (2018).
        Inf2vec: Latent Representation Model for Social Influence Embedding.
        IEEE 34th International Conference on Data Engineering (ICDE), 946-957.

    Attributes:
        graph: 输入图对象。
        dimensions: 嵌入维度。
        source_embeddings: 源嵌入矩阵（影响能力）。
        target_embeddings: 目标嵌入矩阵（被影响趋势）。
        embeddings: 融合后的嵌入矩阵（Source + Target 拼接）。

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.embedding import Inf2Vec
        >>>
        >>> graph = IMGraph(edges, weights=0.3)
        >>> cascades = [[0, 1, 3], [2, 4, 5, 1]]
        >>> model = Inf2Vec(graph, cascades, dimensions=64)
        >>> model.train()
        >>> source_emb = model.get_source_embedding(0)
        >>> target_emb = model.get_target_embedding(0)
    """

    def __init__(
        self,
        graph: 'IMGraph',
        cascades: Optional[List[List[int]]] = None,
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        restart_prob: float = 0.5,
        alpha: float = 0.5,
        random_seed: Optional[int] = None,
    ):
        """初始化 Inf2Vec 模型。

        Args:
            graph: IMGraph 图对象。
            cascades: 扩散级联列表，每个级联是一个节点 ID 列表，
                      按时间顺序排列（第一个为发起者）。若为 None，
                      则仅使用网络结构生成上下文。默认 None。
            dimensions: 嵌入维度，默认 128。
            walk_length: 每次随机游走的长度，默认 80。
            num_walks: 每个节点的游走次数，默认 10。
            restart_prob: 随机游走重启概率，默认 0.5。
            alpha: 局部影响上下文与全局相似性上下文的混合权重，
                   0 表示仅使用局部，1 表示仅使用全局。默认 0.5。
            random_seed: 随机种子，默认 None（使用全局种子）。
        """
        super().__init__(graph, dimensions, walk_length, num_walks, random_seed)

        self.cascades = cascades or []
        self.restart_prob = restart_prob
        self.alpha = alpha

        self.source_embeddings: Optional[np.ndarray] = None
        self.target_embeddings: Optional[np.ndarray] = None
        self.embeddings: Optional[np.ndarray] = None

        self._propagation_nets: List[Dict[int, List[int]]] = []
        self._user_similarity: Dict[int, List[int]] = {}

    def _extract_propagation_networks(self) -> None:
        """从扩散级联中提取影响传播网络。

        对于每个级联，根据网络拓扑和时间顺序，推断节点间的影响关系。
        如果级联中节点 j 在节点 i 之后被激活，且网络中存在 i -> j 的边，
        则认为 i 对 j 有影响。
        """
        self._propagation_nets = []

        for cascade in self.cascades:
            if len(cascade) < 2:
                continue

            active_set = set()
            prop_net: Dict[int, List[int]] = defaultdict(list)

            for i, node_u in enumerate(cascade):
                active_set.add(node_u)
                for node_v in cascade[i + 1:]:
                    if self.graph.has_edge(node_u, node_v):
                        if node_v not in prop_net[node_u]:
                            prop_net[node_u].append(node_v)

            self._propagation_nets.append(dict(prop_net))

    def _compute_user_similarity(self) -> None:
        """计算全局用户相似性上下文。

        基于扩散级联中节点的共现关系，构建用户兴趣相似性。
        两个节点在同一个级联中出现得越多，它们的兴趣越相似。
        """
        co_occurrence: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for cascade in self.cascades:
            if len(cascade) < 2:
                continue
            cascade_set = list(set(cascade))
            for i in range(len(cascade_set)):
                for j in range(i + 1, len(cascade_set)):
                    u, v = cascade_set[i], cascade_set[j]
                    co_occurrence[u][v] += 1
                    co_occurrence[v][u] += 1

        self._user_similarity = {}
        for node in range(self.graph.num_nodes):
            if node in co_occurrence:
                sorted_sim = sorted(
                    co_occurrence[node].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                self._user_similarity[node] = [n for n, _ in sorted_sim]

    def _do_rwr(
        self,
        start_node: int,
        prop_net: Dict[int, List[int]],
    ) -> List[int]:
        """在影响传播网络上执行带重启的随机游走。

        Args:
            start_node: 起始节点。
            prop_net: 影响传播网络（邻接表）。

        Returns:
            List[int]: 游走路径。
        """
        walk = [start_node]
        current = start_node

        for _ in range(self.walk_length - 1):
            if random.random() < self.restart_prob:
                current = start_node
            else:
                neighbors = prop_net.get(current, [])
                if len(neighbors) == 0:
                    neighbors = list(self.graph.out_neighbors(current))

                if len(neighbors) == 0:
                    break

                current = random.choice(neighbors)

            walk.append(current)

        return walk

    def _generate_local_contexts(self) -> List[List[int]]:
        """生成局部影响上下文（从传播网络上的随机游走）。

        Returns:
            List[List[int]]: 游走序列列表。
        """
        walks = []

        for _ in range(self.num_walks):
            for prop_net in self._propagation_nets:
                all_nodes = set()
                for u, neighbors in prop_net.items():
                    all_nodes.add(u)
                    all_nodes.update(neighbors)

                nodes = list(all_nodes)
                random.shuffle(nodes)

                for node in nodes:
                    walk = self._do_rwr(node, prop_net)
                    if len(walk) > 1:
                        walks.append(walk)

        return walks

    def _generate_global_contexts(self) -> List[List[int]]:
        """生成全局用户相似性上下文。

        对于每个节点，将其与相似节点组成上下文对。

        Returns:
            List[List[int]]: 上下文序列列表。
        """
        walks = []

        for node, similar_nodes in self._user_similarity.items():
            if len(similar_nodes) == 0:
                continue

            for _ in range(self.num_walks):
                walk = [node]
                remaining = self.walk_length - 1
                sample_size = min(remaining, len(similar_nodes))
                sampled = random.sample(similar_nodes, sample_size)
                walk.extend(sampled)

                if len(walk) > 1:
                    walks.append(walk)

        return walks

    def _do_walk(self, start_node: int) -> List[int]:
        """从指定节点开始执行随机游走（使用图结构作为后备）。

        当没有级联数据时，退化为在原始图上的随机游走。

        Args:
            start_node: 起始节点。

        Returns:
            List[int]: 游走路径。
        """
        walk = [start_node]

        for _ in range(self.walk_length - 1):
            current = walk[-1]
            neighbors = list(self.graph.out_neighbors(current))

            if len(neighbors) == 0:
                break

            if random.random() < self.restart_prob:
                walk.append(start_node)
            else:
                walk.append(random.choice(neighbors))

        return walk

    def _generate_walks(self) -> None:
        """生成所有随机游走序列。

        结合局部影响上下文和全局用户相似性上下文。
        """
        self._walks = []

        if len(self._propagation_nets) > 0:
            local_walks = self._generate_local_contexts()
            self._walks.extend(local_walks)

            if self.alpha < 1.0:
                global_walks = self._generate_global_contexts()
                n_local = len(local_walks)
                n_global = len(global_walks)

                if n_local > 0 and n_global > 0:
                    n_keep_global = int(n_local * self.alpha / (1.0 - self.alpha + 1e-10))
                    n_keep_global = min(n_keep_global, n_global)
                    if n_keep_global > 0:
                        self._walks.extend(random.sample(global_walks, n_keep_global))
                elif n_local == 0 and n_global > 0:
                    self._walks.extend(global_walks)
        else:
            for _ in range(self.num_walks):
                nodes = list(range(self.graph.num_nodes))
                random.shuffle(nodes)
                for node in nodes:
                    walk = self._do_walk(node)
                    if len(walk) > 1:
                        self._walks.append(walk)

    def train(
        self,
        window_size: int = 10,
        min_count: int = 1,
        epochs: int = 5,
        learning_rate: float = 0.025,
        negative_samples: int = 5,
        backend: str = 'auto',
    ) -> None:
        """训练 Inf2Vec 模型。

        Args:
            window_size: 上下文窗口大小，默认 10。
            min_count: 忽略出现次数少于此值的节点，默认 1。
            epochs: 训练轮数，默认 5。
            learning_rate: 初始学习率，默认 0.025。
            negative_samples: 负采样数量，默认 5。
            backend: 后端实现，支持 'auto'、'gensim'、'pytorch'。
                     'auto' 优先使用 gensim，不可用时使用 pytorch。默认 'auto'。
        """
        self._extract_propagation_networks()

        if len(self.cascades) > 0:
            self._compute_user_similarity()

        self._generate_walks()

        if backend == 'auto':
            try:
                import gensim
                backend = 'gensim'
            except ImportError:
                backend = 'pytorch'

        if backend == 'gensim':
            self._train_gensim_inf2vec(
                window_size, min_count, epochs, learning_rate, negative_samples
            )
        elif backend == 'pytorch':
            self._train_pytorch_inf2vec(
                window_size, epochs, learning_rate, negative_samples
            )
        else:
            raise ValueError(
                f"不支持的后端: {backend}，请选择 'gensim' 或 'pytorch'"
            )

    def _train_gensim_inf2vec(
        self,
        window_size: int,
        min_count: int,
        epochs: int,
        learning_rate: float,
        negative_samples: int,
    ) -> None:
        """使用 gensim 训练 Inf2Vec（Source/Target 分离嵌入）。

        利用 Word2Vec 的输入权重作为 Source embedding，
        输出权重作为 Target embedding。
        """
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

        num_nodes = self.graph.num_nodes
        self.source_embeddings = np.zeros((num_nodes, self.dimensions))
        self.target_embeddings = np.zeros((num_nodes, self.dimensions))

        for node in range(num_nodes):
            node_str = str(node)
            if node_str in model.wv:
                self.source_embeddings[node] = model.wv[node_str]

        try:
            for node in range(num_nodes):
                node_str = str(node)
                if node_str in model.wv.key_to_index:
                    idx = model.wv.key_to_index[node_str]
                    self.target_embeddings[node] = model.wv.vectors_norm[idx] if hasattr(model.wv, 'vectors_norm') else model.syn1neg[idx] if hasattr(model, 'syn1neg') else model.wv[node_str]
        except (AttributeError, KeyError):
            self.target_embeddings = self.source_embeddings.copy()

        self.embeddings = np.concatenate(
            [self.source_embeddings, self.target_embeddings], axis=1
        )

    def _train_pytorch_inf2vec(
        self,
        window_size: int,
        epochs: int,
        learning_rate: float,
        negative_samples: int,
    ) -> None:
        """使用 PyTorch 训练 Inf2Vec（Source/Target 分离嵌入）。

        实现 NCE 损失，Source embedding 和 Target embedding 分别对应
        Skip-gram 的输入和输出权重矩阵。
        """
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
        embed_dim = self.dimensions

        class Inf2VecModel(nn.Module):
            def __init__(self, vocab_size, embed_dim):
                super().__init__()
                self.source_embed = nn.Embedding(vocab_size, embed_dim)
                self.target_embed = nn.Embedding(vocab_size, embed_dim)
                self.source_bias = nn.Parameter(torch.zeros(vocab_size))
                self.target_bias = nn.Parameter(torch.zeros(vocab_size))

                nn.init.uniform_(self.source_embed.weight, -1.0 / embed_dim, 1.0 / embed_dim)
                std = 1.0 / math.sqrt(embed_dim)
                nn.init.normal_(self.target_embed.weight, 0, std)

            def forward(self, source, target, neg_targets):
                source_emb = self.source_embed(source)
                target_emb = self.target_embed(target)
                neg_emb = self.target_embed(neg_targets)

                source_bias = self.source_bias[source]
                target_bias = self.target_bias[target]

                pos_score = torch.sum(source_emb * target_emb, dim=1) + source_bias + target_bias
                pos_score = torch.clamp(pos_score, -10, 10)
                pos_loss = -torch.nn.functional.logsigmoid(pos_score)

                neg_bias = self.target_bias[neg_targets]
                neg_score = torch.bmm(neg_emb, source_emb.unsqueeze(2)).squeeze(2) + neg_bias
                neg_score = torch.clamp(neg_score, -10, 10)
                neg_loss = -torch.nn.functional.logsigmoid(-neg_score)
                neg_loss = torch.sum(neg_loss, dim=1)

                return torch.mean(pos_loss + neg_loss)

        training_data = []
        for walk in self._walks:
            for i, source in enumerate(walk):
                start = max(0, i - window_size)
                end = min(len(walk), i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        training_data.append((source, walk[j]))

        if len(training_data) == 0:
            warnings.warn("没有生成训练数据，请检查图结构或级联数据。")
            self.source_embeddings = np.zeros((vocab_size, embed_dim))
            self.target_embeddings = np.zeros((vocab_size, embed_dim))
            self.embeddings = np.zeros((vocab_size, embed_dim * 2))
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Inf2VecModel(vocab_size, embed_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        batch_size = 512
        for epoch in range(epochs):
            random.shuffle(training_data)
            total_loss = 0.0
            num_batches = 0

            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]

                sources = torch.tensor(
                    [x[0] for x in batch], dtype=torch.long, device=device
                )
                targets = torch.tensor(
                    [x[1] for x in batch], dtype=torch.long, device=device
                )
                neg_targets = torch.randint(
                    0, vocab_size,
                    (len(batch), negative_samples),
                    dtype=torch.long,
                    device=device,
                )

                optimizer.zero_grad()
                loss = model(sources, targets, neg_targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.source_embeddings = model.source_embed.weight.detach().cpu().numpy()
        self.target_embeddings = model.target_embed.weight.detach().cpu().numpy()
        self.embeddings = np.concatenate(
            [self.source_embeddings, self.target_embeddings], axis=1
        )

    def get_source_embedding(self, node: int) -> np.ndarray:
        """获取指定节点的源嵌入向量（影响能力）。

        Args:
            node: 节点 ID。

        Returns:
            np.ndarray: 节点的源嵌入向量。

        Raises:
            ValueError: 模型尚未训练。
        """
        if self.source_embeddings is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法。")
        return self.source_embeddings[node]

    def get_target_embedding(self, node: int) -> np.ndarray:
        """获取指定节点的目标嵌入向量（被影响趋势）。

        Args:
            node: 节点 ID。

        Returns:
            np.ndarray: 节点的目标嵌入向量。

        Raises:
            ValueError: 模型尚未训练。
        """
        if self.target_embeddings is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法。")
        return self.target_embeddings[node]

    def get_influence_probability(self, source: int, target: int) -> float:
        """计算节点 source 对节点 target 的影响概率。

        使用 softmax 归一化的点积计算:
        P(target | source) = exp(S_source · T_target) / Z

        Args:
            source: 源节点 ID。
            target: 目标节点 ID。

        Returns:
            float: 影响概率。

        Raises:
            ValueError: 模型尚未训练。
        """
        if self.source_embeddings is None or self.target_embeddings is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法。")

        s_emb = self.source_embeddings[source]
        t_emb = self.target_embeddings[target]

        dot_product = np.dot(s_emb, t_emb)

        neighbors = list(self.graph.out_neighbors(source))
        if len(neighbors) == 0:
            return 0.0

        all_dots = []
        for neighbor in neighbors:
            t_neighbor = self.target_embeddings[neighbor]
            all_dots.append(np.dot(s_emb, t_neighbor))

        max_dot = max(all_dots)
        exp_dots = [math.exp(d - max_dot) for d in all_dots]
        sum_exp = sum(exp_dots)

        target_dot = dot_product
        if target in neighbors:
            idx = neighbors.index(target)
            return exp_dots[idx] / sum_exp
        else:
            return math.exp(target_dot - max_dot) / sum_exp

    def get_influence_weights(self) -> Dict[Tuple[int, int], float]:
        """计算所有边的影响权重。

        使用 S_u · T_v 计算每条边的影响权重，并按出度归一化。

        Returns:
            Dict[Tuple[int, int], float]: 边到影响权重的映射。

        Raises:
            ValueError: 模型尚未训练。
        """
        if self.source_embeddings is None or self.target_embeddings is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法。")

        raw_weights: Dict[int, Dict[int, float]] = defaultdict(dict)

        for u in range(self.graph.num_nodes):
            for v in self.graph.out_neighbors(u):
                weight = np.dot(self.source_embeddings[u], self.target_embeddings[v])
                raw_weights[u][v] = weight

        result: Dict[Tuple[int, int], float] = {}
        for u, targets in raw_weights.items():
            total = sum(targets.values())
            if total == 0:
                continue
            for v, w in targets.items():
                result[(u, v)] = w / total

        return result

    def get_embedding(self, node: int) -> np.ndarray:
        """获取指定节点的融合嵌入向量（Source + Target 拼接）。

        Args:
            node: 节点 ID。

        Returns:
            np.ndarray: 节点的融合嵌入向量，维度为 2 * dimensions。
        """
        if self.embeddings is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法。")
        return self.embeddings[node]

    def get_all_embeddings(self) -> np.ndarray:
        """获取所有节点的融合嵌入矩阵。

        Returns:
            np.ndarray: 融合嵌入矩阵，形状为 (num_nodes, 2 * dimensions)。
        """
        if self.embeddings is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法。")
        return self.embeddings

    def most_similar(self, node: int, topn: int = 10) -> List[tuple]:
        """找出与指定节点最相似的节点（基于源嵌入）。

        Args:
            node: 目标节点 ID。
            topn: 返回的相似节点数量，默认 10。

        Returns:
            List[tuple]: 相似节点列表，每个元素为 (node_id, similarity)。
        """
        if self.source_embeddings is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法。")

        node_vec = self.source_embeddings[node]

        norms = np.linalg.norm(self.source_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = self.source_embeddings / norms

        node_norm = node_vec / (np.linalg.norm(node_vec) + 1e-10)
        similarities = np.dot(normalized, node_norm)

        most_similar_indices = np.argsort(similarities)[::-1][1:topn + 1]

        return [(int(idx), float(similarities[idx])) for idx in most_similar_indices]
