"""Struc2Vec 图嵌入算法。"""

import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pynetim.graph import IMGraph

from .base import BaseEmbedding


def _dtw_distance(seq_a: list, seq_b: list, dist_func=None) -> float:
    """计算两个序列之间的 DTW 距离。

    使用动态规划实现，支持 compact（度-频率对）和普通（度列表）两种格式。

    Args:
        seq_a: 序列 A。
        seq_b: 序列 B。
        dist_func: 元素间距离函数，默认使用 _cost_max。

    Returns:
        float: DTW 距离。
    """
    if dist_func is None:
        dist_func = _cost_max

    n = len(seq_a)
    m = len(seq_b)
    if n == 0 or m == 0:
        return float('inf')

    dtw = np.full((n + 1, m + 1), float('inf'))
    dtw[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_func(seq_a[i - 1], seq_b[j - 1])
            dtw[i][j] = cost + min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])

    return float(dtw[n][m])


def _cost(a: float, b: float) -> float:
    """计算两个度值之间的距离（普通模式）。

    Args:
        a: 度值 a。
        b: 度值 b。

    Returns:
        float: 距离值。
    """
    ep = 0.5
    m = max(a, b) + ep
    mi = min(a, b) + ep
    return (m / mi) - 1


def _cost_max(a: tuple, b: tuple) -> float:
    """计算两个度-频率对之间的距离（compact 模式）。

    Args:
        a: (degree, frequency) 对。
        b: (degree, frequency) 对。

    Returns:
        float: 距离值。
    """
    ep = 0.5
    m = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * max(a[1], b[1])


class Struc2Vec(BaseEmbedding):
    """Struc2Vec 图节点嵌入算法。

    基于结构相似性的图嵌入算法。通过构建多层上下文图捕获节点的结构等价性，
    两个节点即使网络距离很远，只要局部结构相似就会得到相似的嵌入。

    算法流程:
        1. 对每个节点计算各层邻域的有序度序列
        2. 使用 DTW 计算节点对之间的结构距离
        3. 构建多层上下文图（每层连接结构相似的节点对）
        4. 在多层图上进行随机游走（同层移动 + 跨层切换）
        5. 使用 Skip-gram 训练节点嵌入

    Attributes:
        graph: 输入图对象。
        dimensions: 嵌入维度。
        stay_prob: 同层停留概率，默认 0.3。
        opt1_compact: 是否使用 compact 度序列优化，默认 True。
        opt2_reduce_sim_calc: 是否减少相似度计算量，默认 True。
        opt3_num_layers: 最大计算层数，None 表示自动确定，默认 None。
        embeddings: 学习到的节点嵌入矩阵。

    References:
        Ribeiro, L. F. R., Saverese, P. H. P., & Figueiredo, D. R. (2017).
        struc2vec: Learning Node Representations from Structural Identity.
        KDD, 385-394.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.embedding import Struc2Vec
        >>>
        >>> graph = IMGraph(edges, weights=0.3)
        >>> model = Struc2Vec(graph, dimensions=64)
        >>> model.train()
        >>> embeddings = model.get_all_embeddings()
    """

    def __init__(
        self,
        graph: 'IMGraph',
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        stay_prob: float = 0.3,
        opt1_compact: bool = True,
        opt2_reduce_sim_calc: bool = True,
        opt3_num_layers: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        """初始化 Struc2Vec 模型。

        Args:
            graph: IMGraph 图对象。
            dimensions: 嵌入维度，默认 128。
            walk_length: 每次随机游走的长度，默认 80。
            num_walks: 每个节点的游走次数，默认 10。
            stay_prob: 同层停留概率（1 - stay_prob 为跨层切换概率），默认 0.3。
            opt1_compact: 是否使用 compact 度序列（度-频率对）减少 DTW 计算量，默认 True。
            opt2_reduce_sim_calc: 是否只计算度相近的节点对之间的距离，默认 True。
            opt3_num_layers: 最大计算层数，None 表示自动确定（图的直径），默认 None。
            random_seed: 随机种子，默认 None（使用全局种子）。
        """
        super().__init__(graph, dimensions, walk_length, num_walks, random_seed)

        self.stay_prob = stay_prob
        self.opt1_compact = opt1_compact
        self.opt2_reduce_sim_calc = opt2_reduce_sim_calc
        self.opt3_num_layers = opt3_num_layers

        self._degree_list: Dict[int, Dict[int, list]] = {}
        self._distances: Dict[Tuple[int, int], Dict[int, float]] = {}
        self._layers_adj: Dict[int, Dict[int, List[int]]] = {}
        self._layers_distances: Dict[int, Dict[Tuple[int, int], float]] = {}
        self._layers_alias: Dict[int, Dict[int, np.ndarray]] = {}
        self._layers_accept: Dict[int, Dict[int, np.ndarray]] = {}
        self._average_weight: Dict[int, float] = {}
        self._gamma: Dict[int, Dict[int, int]] = {}

    def _compute_ordered_degree_list(self) -> None:
        """对每个节点计算各层邻域的有序度序列。"""
        self._degree_list = {}
        num_nodes = self.graph.num_nodes

        for v in range(num_nodes):
            self._degree_list[v] = self._get_ordered_degree_list_node(v)

    def _get_ordered_degree_list_node(self, root: int) -> Dict[int, list]:
        """使用 BFS 计算单个节点各层邻域的有序度序列。

        Args:
            root: 根节点。

        Returns:
            Dict[int, list]: 各层的有序度序列。
        """
        max_layer = self.opt3_num_layers if self.opt3_num_layers is not None else float('inf')
        ordered_degree_dict = {}
        visited = [False] * self.graph.num_nodes
        queue = deque()
        level = 0

        queue.append(root)
        visited[root] = True

        while queue and level <= max_layer:
            count = len(queue)

            if self.opt1_compact:
                degree_freq = {}
            else:
                degree_list = []

            while count > 0:
                node = queue.popleft()
                degree = len(list(self.graph.out_neighbors(node)))

                if self.opt1_compact:
                    degree_freq[degree] = degree_freq.get(degree, 0) + 1
                else:
                    degree_list.append(degree)

                for neighbor in self.graph.out_neighbors(node):
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

                count -= 1

            if self.opt1_compact:
                ordered_list = sorted([(d, f) for d, f in degree_freq.items()])
            else:
                ordered_list = sorted(degree_list)

            ordered_degree_dict[level] = ordered_list
            level += 1

        return ordered_degree_dict

    def _create_degree_vectors(self) -> Dict[int, dict]:
        """创建度向量用于优化相似度计算。

        Returns:
            Dict[int, dict]: 度向量字典。
        """
        degrees = {}
        degrees_sorted = set()
        num_nodes = self.graph.num_nodes

        for v in range(num_nodes):
            degree = len(list(self.graph.out_neighbors(v)))
            degrees_sorted.add(degree)
            if degree not in degrees:
                degrees[degree] = {}
                degrees[degree]['vertices'] = []
            degrees[degree]['vertices'].append(v)

        degrees_sorted = sorted(degrees_sorted)

        for index, degree in enumerate(degrees_sorted):
            if index > 0:
                degrees[degree]['before'] = degrees_sorted[index - 1]
            if index < len(degrees_sorted) - 1:
                degrees[degree]['after'] = degrees_sorted[index + 1]

        return degrees

    def _get_vertices(
        self, v: int, degree_v: int, degrees: Dict[int, dict], num_nodes: int
    ) -> List[int]:
        """获取与节点 v 度相近的节点列表（优化：减少 DTW 计算量）。

        Args:
            v: 目标节点。
            degree_v: 目标节点的度。
            degrees: 度向量字典。
            num_nodes: 图中节点总数。

        Returns:
            List[int]: 候选节点列表。
        """
        max_selected = 2 * math.log(num_nodes, 2)
        vertices = []
        c_v = 0

        try:
            for v2 in degrees[degree_v]['vertices']:
                if v != v2:
                    vertices.append(v2)
                    c_v += 1
                    if c_v > max_selected:
                        raise StopIteration

            degree_b = degrees[degree_v].get('before', -1)
            degree_a = degrees[degree_v].get('after', -1)

            if degree_b == -1 and degree_a == -1:
                raise StopIteration

            degree_now = self._verify_degrees(degrees, degree_v, degree_a, degree_b)

            while True:
                for v2 in degrees[degree_now]['vertices']:
                    if v != v2:
                        vertices.append(v2)
                        c_v += 1
                        if c_v > max_selected:
                            raise StopIteration

                if degree_now == degree_b:
                    degree_b = degrees.get(degree_b, {}).get('before', -1)
                else:
                    degree_a = degrees.get(degree_a, {}).get('after', -1)

                if degree_b == -1 and degree_a == -1:
                    raise StopIteration

                degree_now = self._verify_degrees(degrees, degree_v, degree_a, degree_b)

        except StopIteration:
            pass

        return vertices

    @staticmethod
    def _verify_degrees(
        degrees: Dict[int, dict], degree_v_root: int, degree_a: int, degree_b: int
    ) -> int:
        """选择距离根节点度更近的相邻度。

        Args:
            degrees: 度向量字典。
            degree_v_root: 根节点度。
            degree_a: 较大的相邻度。
            degree_b: 较小的相邻度。

        Returns:
            int: 选择的度值。
        """
        if degree_b == -1:
            return degree_a
        elif degree_a == -1:
            return degree_b
        elif abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root):
            return degree_b
        else:
            return degree_a

    def _compute_structural_distances(self) -> None:
        """计算节点对之间的结构距离。"""
        self._compute_ordered_degree_list()

        dist_func = _cost_max if self.opt1_compact else _cost

        if self.opt2_reduce_sim_calc:
            degrees = self._create_degree_vectors()
            num_nodes = self.graph.num_nodes
            vertices_to_compare = {}
            for v in range(num_nodes):
                degree_v = len(list(self.graph.out_neighbors(v)))
                vertices_to_compare[v] = self._get_vertices(v, degree_v, degrees, num_nodes)
        else:
            vertices_to_compare = {}
            for v in range(self.graph.num_nodes):
                vertices_to_compare[v] = [vd for vd in range(self.graph.num_nodes) if vd > v]

        self._distances = {}
        for v1, nbs in vertices_to_compare.items():
            lists_v1 = self._degree_list[v1]
            for v2 in nbs:
                lists_v2 = self._degree_list[v2]
                max_layer = min(len(lists_v1), len(lists_v2))
                self._distances[v1, v2] = {}

                for layer in range(max_layer):
                    dist = _dtw_distance(lists_v1[layer], lists_v2[layer], dist_func)
                    self._distances[v1, v2][layer] = dist

        self._consolidate_distances()

    def _consolidate_distances(self, start_layer: int = 1) -> None:
        """累积结构距离：第 k 层的距离 = 第 k 层原始距离 + 第 k-1 层累积距离。

        Args:
            start_layer: 开始累积的层，默认 1。
        """
        for vertices, layers in self._distances.items():
            keys_layers = sorted(layers.keys())
            actual_start = min(len(keys_layers), start_layer)
            for _ in range(actual_start):
                keys_layers.pop(0)

            for layer in keys_layers:
                layers[layer] += layers[layer - 1]

    def _create_context_graph(self) -> None:
        """构建多层上下文图并计算转移概率。"""
        layer_distances = {}
        layer_adj = {}

        for v_pair, layer_dist in self._distances.items():
            for layer, distance in layer_dist.items():
                vx = v_pair[0]
                vy = v_pair[1]

                layer_distances.setdefault(layer, {})
                layer_distances[layer][vx, vy] = distance

                layer_adj.setdefault(layer, {})
                layer_adj[layer].setdefault(vx, [])
                layer_adj[layer].setdefault(vy, [])
                layer_adj[layer][vx].append(vy)
                layer_adj[layer][vy].append(vx)

        self._layers_adj = layer_adj
        self._layers_distances = layer_distances

        self._compute_transition_probs()

    def _compute_transition_probs(self) -> None:
        """计算多层图中同层移动的转移概率（使用别名采样）。"""
        self._layers_alias = {}
        self._layers_accept = {}
        self._average_weight = {}
        self._gamma = {}

        for layer in self._layers_adj:
            neighbors = self._layers_adj[layer]
            layer_distances = self._layers_distances[layer]
            node_alias_dict = {}
            node_accept_dict = {}
            norm_weights = {}

            sum_weights = 0.0
            sum_edges = 0

            for v, v_neighbors in neighbors.items():
                e_list = []
                sum_w = 0.0

                for n in v_neighbors:
                    wd = layer_distances.get((v, n), layer_distances.get((n, v), 0))
                    w = math.exp(-float(wd))
                    e_list.append(w)
                    sum_w += w

                e_list = [x / sum_w for x in e_list]
                norm_weights[v] = e_list
                accept, alias = self._create_alias_table(e_list)
                node_alias_dict[v] = alias
                node_accept_dict[v] = accept

                sum_weights += sum_w
                sum_edges += len(v_neighbors)

            self._average_weight[layer] = sum_weights / max(sum_edges, 1)

            self._gamma[layer] = {}
            for v, list_weights in norm_weights.items():
                num_neighbours = sum(1 for w in list_weights if w > self._average_weight[layer])
                self._gamma[layer][v] = num_neighbours

            self._layers_alias[layer] = node_alias_dict
            self._layers_accept[layer] = node_accept_dict

    def _create_alias_table(self, probs: list) -> Tuple[np.ndarray, np.ndarray]:
        """创建别名采样表。

        Args:
            probs: 概率分布列表。

        Returns:
            Tuple[np.ndarray, np.ndarray]: (accept, alias) 数组。
        """
        n = len(probs)
        accept = np.zeros(n, dtype=np.float64)
        alias = np.zeros(n, dtype=np.int32)

        scaled = np.array(probs) * n
        small = []
        large = []

        for i, p in enumerate(scaled):
            if p < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            small_idx = small.pop()
            large_idx = large.pop()

            accept[small_idx] = scaled[small_idx]
            alias[small_idx] = large_idx

            scaled[large_idx] = scaled[large_idx] + scaled[small_idx] - 1.0

            if scaled[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        while large:
            accept[large.pop()] = 1.0

        while small:
            accept[small.pop()] = 1.0

        return accept, alias

    def _alias_draw(self, layer: int, v: int) -> int:
        """使用别名采样从指定层和节点的邻居中采样。

        Args:
            layer: 层号。
            v: 节点。

        Returns:
            int: 采样得到的邻居节点。
        """
        neighbors = self._layers_adj[layer][v]
        n = len(neighbors)

        kk = random.randint(0, n - 1)
        if random.random() < self._layers_accept[layer][v][kk]:
            return neighbors[kk]
        else:
            return neighbors[self._layers_alias[layer][v][kk]]

    def _prob_move_up(self, amount_neighbours: int) -> float:
        """计算上移一层的概率。

        Args:
            amount_neighbours: 当前节点在当前层中权重大于平均权重的邻居数。

        Returns:
            float: 上移概率。
        """
        x = math.log(amount_neighbours + math.e)
        return x / (x + 1)

    def _do_walk(self, start_node: int) -> List[int]:
        """从指定节点开始在多层图上执行一次随机游走。

        游走策略:
            - 以 stay_prob 概率在同层移动（选择结构相似的邻居）
            - 以 (1 - stay_prob) 概率跨层切换：
                - 根据邻居数量决定上移或下移
                - 上移到更大范围的邻域结构层
                - 下移到更小范围的邻域结构层

        Args:
            start_node: 起始节点。

        Returns:
            List[int]: 游走路径。
        """
        v = start_node
        layer = 0
        path = [v]

        while len(path) < self.walk_length:
            r = random.random()

            if r < self.stay_prob:
                if layer in self._layers_adj and v in self._layers_adj[layer]:
                    v = self._alias_draw(layer, v)
                path.append(v)
            else:
                r = random.random()
                gamma_v = self._gamma.get(layer, {}).get(v, 1)
                move_up_prob = self._prob_move_up(gamma_v)

                if r > move_up_prob:
                    if layer > 0:
                        layer -= 1
                else:
                    if (layer + 1) in self._layers_adj and v in self._layers_adj[layer + 1]:
                        layer += 1

        return path

    def train(
        self,
        window_size: int = 10,
        min_count: int = 1,
        epochs: int = 5,
        learning_rate: float = 0.025,
        negative_samples: int = 5,
        backend: str = 'auto',
    ) -> None:
        """训练 Struc2Vec 模型。

        Args:
            window_size: 上下文窗口大小，默认 10。
            min_count: 忽略出现次数少于此值的节点，默认 1。
            epochs: 训练轮数，默认 5。
            learning_rate: 初始学习率，默认 0.025。
            negative_samples: 负采样数量，默认 5。
            backend: 后端实现，支持 'auto'、'gensim'、'pytorch'。
                     'auto' 优先使用 gensim，不可用时使用 pytorch。默认 'auto'。
        """
        self._compute_structural_distances()
        self._create_context_graph()
        self._generate_walks()
        self._train_skipgram(window_size, min_count, epochs, learning_rate, negative_samples, backend)
