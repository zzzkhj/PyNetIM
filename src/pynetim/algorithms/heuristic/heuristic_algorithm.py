import heapq
from typing import List, Set, Dict, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from ...graph import IMGraph

from ..base_algorithm import BaseAlgorithm


class DegreeCentralityAlgorithm(BaseAlgorithm):
    """度中心性启发式算法。

    选择出度最大的节点作为种子节点，是最简单快速的启发式方法。
    适合作为基准算法进行比较。

    时间复杂度: O(n log n)

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import DegreeCentralityAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = DegreeCentralityAlgorithm(graph)
        >>> seeds = algo.run(k=10)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = None):
        super().__init__(graph, diffusion_model)

    def run(self, k: int) -> Set[int]:
        degrees = [(self.graph.out_degree(v), v) for v in range(self.graph.num_nodes)]
        degrees.sort(reverse=True)
        seeds = {v for _, v in degrees[:k]}
        self.seeds = seeds
        return seeds


class PageRankAlgorithm(BaseAlgorithm):
    """PageRank 启发式算法。

    基于 Google 的 PageRank 算法，通过模拟随机游走来评估节点的重要性。
    节点的重要性取决于指向它的节点的重要性。

    时间复杂度: O(n * max_iter)

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。

    References:
        Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual 
        Web search engine. Computer networks and ISDN systems, 30(1-7), 107-117.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import PageRankAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = PageRankAlgorithm(graph)
        >>> seeds = algo.run(k=10)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = None, 
                 damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        """初始化 PageRank 算法。

        Args:
            graph: 输入图对象。
            diffusion_model: 扩散模型（此算法不使用该参数）。
            damping: 阻尼系数，表示继续随机游走的概率，默认 0.85。
            max_iter: 最大迭代次数，默认 100。
            tol: 收敛阈值，默认 1e-6。
        """
        super().__init__(graph, diffusion_model)
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    def run(self, k: int) -> Set[int]:
        n = self.graph.num_nodes
        pr = {v: 1.0 / n for v in range(n)}
        
        in_neighbors: Dict[int, List[tuple]] = defaultdict(list)
        out_degree = {}
        
        for v in range(n):
            neighbors = list(self.graph.out_neighbors_with_weights(v))
            out_degree[v] = len(neighbors)
            for u, w in neighbors:
                in_neighbors[u].append((v, w))
        
        for _ in range(self.max_iter):
            new_pr = {}
            diff = 0.0
            
            for v in range(n):
                rank_sum = 0.0
                for u, w in in_neighbors[v]:
                    if out_degree[u] > 0:
                        rank_sum += pr[u] / out_degree[u]
                
                new_pr[v] = (1 - self.damping) / n + self.damping * rank_sum
                diff += abs(new_pr[v] - pr[v])
            
            pr = new_pr
            if diff < self.tol:
                break
        
        ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        seeds = {v for v, _ in ranked[:k]}
        self.seeds = seeds
        return seeds


class VoteRankAlgorithm(BaseAlgorithm):
    """VoteRank 启发式算法。

    通过投票机制选择分散的影响力节点。每个节点为其邻居投票，
    得票最高的节点被选为种子，然后其邻居的投票能力被削弱。
    这样可以避免选择过于聚集的种子节点。

    时间复杂度: O(n * k)

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。

    References:
        Zhang, J. X., Chen, D. B., Dong, Q., & Zhao, Z. D. (2016). 
        Identifying a set of influential spreaders in complex networks 
        by VoteRank. Physica A: Statistical Mechanics and its Applications, 
        461, 171-182.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import VoteRankAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = VoteRankAlgorithm(graph)
        >>> seeds = algo.run(k=10)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = None):
        super().__init__(graph, diffusion_model)

    def run(self, k: int) -> Set[int]:
        n = self.graph.num_nodes
        
        vote_ability = {v: 1.0 for v in range(n)}
        votes = {v: 0.0 for v in range(n)}
        seeds: Set[int] = set()
        
        for _ in range(k):
            votes = {v: 0.0 for v in range(n)}
            
            for v in range(n):
                if vote_ability[v] > 0:
                    for u, _ in self.graph.out_neighbors_with_weights(v):
                        votes[u] += vote_ability[v]
            
            for v in seeds:
                votes[v] = -1
            
            max_vote = max(votes.values())
            if max_vote <= 0:
                break
            
            max_node = max(votes, key=votes.get)
            seeds.add(max_node)
            
            for u, _ in self.graph.out_neighbors_with_weights(max_node):
                vote_ability[u] = max(0, vote_ability[u] - 1.0 / self.graph.out_degree(max_node))
        
        self.seeds = seeds
        return seeds


class KShellDecompositionAlgorithm(BaseAlgorithm):
    """K-shell 分解启发式算法。

    通过迭代移除度数最小的节点，将网络分层。位于核心层（高 k-shell 值）
    的节点被认为具有更大的影响力。

    时间复杂度: O(m)，m 为边数

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。

    References:
        Kitsak, M., Gallos, L. K., Havlin, S., Liljeros, F., Muchnik, L., 
        Stanley, H. E., & Makse, H. A. (2010). Identification of influential 
        spreaders in complex networks. Nature physics, 6(11), 888-893.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import KShellDecompositionAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = KShellDecompositionAlgorithm(graph)
        >>> seeds = algo.run(k=10)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = None):
        super().__init__(graph, diffusion_model)

    def run(self, k: int) -> Set[int]:
        n = self.graph.num_nodes
        
        degree = {v: self.graph.out_degree(v) for v in range(n)}
        k_shell = {v: 0 for v in range(n)}
        
        remaining = set(range(n))
        current_k = 1
        
        while remaining:
            changed = True
            while changed:
                changed = False
                to_remove = []
                
                for v in list(remaining):
                    if degree[v] <= current_k:
                        k_shell[v] = current_k
                        to_remove.append(v)
                        changed = True
                
                for v in to_remove:
                    remaining.remove(v)
                    for u, _ in self.graph.out_neighbors_with_weights(v):
                        if u in remaining:
                            degree[u] -= 1
            
            if remaining:
                current_k += 1
        
        ranked = sorted(k_shell.items(), key=lambda x: (x[1], degree[x[0]]), reverse=True)
        seeds = {v for v, _ in ranked[:k]}
        self.seeds = seeds
        return seeds


class BetweennessCentralityAlgorithm(BaseAlgorithm):
    """介数中心性启发式算法。

    选择在网络中作为最短路径"桥梁"的节点。介数中心性高的节点
    控制着网络中的信息流动，是关键的传播节点。

    时间复杂度: O(n * m) 或 O(n^2) 对于稀疏图

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。

    References:
        Freeman, L. C. (1977). A set of measures of centrality based on 
        betweenness. Sociometry, 35-41.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import BetweennessCentralityAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = BetweennessCentralityAlgorithm(graph)
        >>> seeds = algo.run(k=10)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = None, 
                 normalized: bool = True, sample_size: int = None):
        """初始化介数中心性算法。

        Args:
            graph: 输入图对象。
            diffusion_model: 扩散模型（此算法不使用该参数）。
            normalized: 是否归一化介数值，默认 True。
            sample_size: 采样节点数，用于加速大规模图计算，默认 None（使用全部节点）。
        """
        super().__init__(graph, diffusion_model)
        self.normalized = normalized
        self.sample_size = sample_size

    def run(self, k: int) -> Set[int]:
        import random
        
        n = self.graph.num_nodes
        betweenness = {v: 0.0 for v in range(n)}
        
        nodes = list(range(n))
        if self.sample_size and self.sample_size < n:
            nodes = random.sample(nodes, self.sample_size)
        
        adj = {}
        for v in range(n):
            adj[v] = [u for u, _ in self.graph.out_neighbors_with_weights(v)]
        
        for s in nodes:
            stack = []
            predecessors = {v: [] for v in range(n)}
            sigma = {v: 0 for v in range(n)}
            sigma[s] = 1
            dist = {v: -1 for v in range(n)}
            dist[s] = 0
            
            queue = [s]
            
            while queue:
                v = queue.pop(0)
                stack.append(v)
                
                for w in adj[v]:
                    if dist[w] < 0:
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)
            
            delta = {v: 0.0 for v in range(n)}
            
            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                
                if w != s:
                    betweenness[w] += delta[w]
        
        if self.normalized and n > 2:
            scale = 1.0 / ((n - 1) * (n - 2))
            for v in betweenness:
                betweenness[v] *= scale
        
        ranked = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        seeds = {v for v, _ in ranked[:k]}
        self.seeds = seeds
        return seeds


class ClosenessCentralityAlgorithm(BaseAlgorithm):
    """接近中心性启发式算法。

    选择距离网络中其他节点最近的节点。接近中心性高的节点
    可以快速将信息传播到整个网络。

    时间复杂度: O(n * m)

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。

    References:
        Sabidussi, G. (1966). The centrality index of a graph. 
        Psychometrika, 31(4), 581-603.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import ClosenessCentralityAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = ClosenessCentralityAlgorithm(graph)
        >>> seeds = algo.run(k=10)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = None):
        super().__init__(graph, diffusion_model)

    def _bfs_distances(self, source: int) -> Dict[int, int]:
        from collections import deque
        
        n = self.graph.num_nodes
        dist = {v: -1 for v in range(n)}
        dist[source] = 0
        queue = deque([source])
        
        adj = {}
        for v in range(n):
            adj[v] = [u for u, _ in self.graph.out_neighbors_with_weights(v)]
        
        while queue:
            v = queue.popleft()
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
        
        return dist

    def run(self, k: int) -> Set[int]:
        n = self.graph.num_nodes
        closeness = {}
        
        for v in range(n):
            dist = self._bfs_distances(v)
            reachable = sum(1 for d in dist.values() if d > 0)
            total_dist = sum(d for d in dist.values() if d > 0)
            
            if total_dist > 0:
                closeness[v] = reachable / (total_dist * (n - 1))
            else:
                closeness[v] = 0.0
        
        ranked = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        seeds = {v for v, _ in ranked[:k]}
        self.seeds = seeds
        return seeds


class EigenvectorCentralityAlgorithm(BaseAlgorithm):
    """特征向量中心性启发式算法。

    节点的重要性取决于其邻居的重要性。一个节点如果连接了许多
    重要的节点，那么它自己也更重要。

    时间复杂度: O(n * max_iter)

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。

    References:
        Bonacich, P. (1972). Factoring and weighting approaches to status 
        scores and clique identification. Journal of Mathematical Sociology, 
        2(1), 113-120.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import EigenvectorCentralityAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = EigenvectorCentralityAlgorithm(graph)
        >>> seeds = algo.run(k=10)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = None,
                 max_iter: int = 100, tol: float = 1e-6):
        """初始化特征向量中心性算法。

        Args:
            graph: 输入图对象。
            diffusion_model: 扩散模型（此算法不使用该参数）。
            max_iter: 最大迭代次数，默认 100。
            tol: 收敛阈值，默认 1e-6。
        """
        super().__init__(graph, diffusion_model)
        self.max_iter = max_iter
        self.tol = tol

    def run(self, k: int) -> Set[int]:
        n = self.graph.num_nodes
        
        adj = {}
        for v in range(n):
            adj[v] = [u for u, _ in self.graph.out_neighbors_with_weights(v)]
        
        x = {v: 1.0 for v in range(n)}
        
        for _ in range(self.max_iter):
            x_new = {}
            for v in range(n):
                x_new[v] = sum(x[u] for u in adj[v])
            
            norm = max(abs(v) for v in x_new.values())
            if norm > 0:
                for v in x_new:
                    x_new[v] /= norm
            
            diff = sum(abs(x_new[v] - x[v]) for v in range(n))
            x = x_new
            
            if diff < self.tol:
                break
        
        ranked = sorted(x.items(), key=lambda item: item[1], reverse=True)
        seeds = {v for v, _ in ranked[:k]}
        self.seeds = seeds
        return seeds


class SingleDiscountAlgorithm(BaseAlgorithm):
    """简单度折扣启发式算法。

    通过逐步选择具有最高度数的节点作为种子，并对其邻居节点的度数进行折扣，
    以避免选择过多相互连接的节点。

    该算法速度快，适合大规模图的快速种子选择。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。

    References:
        Chen, W., Wang, Y., & Yang, S. (2009). Efficient influence maximization 
        in social networks. KDD, 199-208.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import SingleDiscountAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = SingleDiscountAlgorithm(graph)
        >>> seeds = algo.run(k=10)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = None):
        super().__init__(graph, diffusion_model)
    
    def run(self, k: int) -> Set[int]:
        d = {v: self.graph.out_degree(v) for v in range(self.graph.num_nodes)}
        seeds: Set[int] = set()
        selected: Set[int] = set()

        heap = [(-d[v], v) for v in range(self.graph.num_nodes)]
        heapq.heapify(heap)

        while len(seeds) < k:
            _, u = heapq.heappop(heap)

            if u in selected:
                continue
            seeds.add(u)
            selected.add(u)

            for v, _ in self.graph.out_neighbors_with_weights(u):
                if v not in selected:
                    d[v] -= 1
                    heapq.heappush(heap, (-d[v], v))

        self.seeds = seeds
        return seeds


class DegreeDiscountAlgorithm(BaseAlgorithm):
    """度折扣启发式算法。

    是 SingleDiscountAlgorithm 的改进版本，考虑了邻居节点之间的影响关系，
    使用更复杂的折扣公式来更好地评估节点的边际影响力。

    该算法速度快且效果较好，适合大规模图的种子选择。

    Attributes:
        graph: 输入图对象。
        seeds: 种子节点集合。

    References:
        Chen, W., Wang, Y., & Yang, S. (2009). Efficient influence maximization 
        in social networks. KDD, 199-208.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.algorithms import DegreeDiscountAlgorithm
        >>> 
        >>> graph = IMGraph(edges, weights=0.3)
        >>> algo = DegreeDiscountAlgorithm(graph, diffusion_model='IC')
        >>> seeds = algo.run(k=10)
    """

    def __init__(self, graph: 'IMGraph', diffusion_model: str = 'IC'):
        super().__init__(graph, diffusion_model)

    def run(self, k: int) -> Set[int]:
        d = {v: self.graph.out_degree(v) for v in range(self.graph.num_nodes)}
        dd = d.copy()
        t = defaultdict(int)
        seeds: Set[int] = set()

        heap = [(-dd[v], v) for v in range(self.graph.num_nodes)]
        heapq.heapify(heap)
        selected: Set[int] = set()

        while len(seeds) < k:
            _, u = heapq.heappop(heap)
            if u in selected:
                continue
            seeds.add(u)
            selected.add(u)

            for v, weight in self.graph.out_neighbors_with_weights(u):
                if v in selected:
                    continue
                t[v] += 1
                dd[v] = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * weight
                heapq.heappush(heap, (-dd[v], v))

        self.seeds = seeds
        return seeds
