"""投票启发式算法。

包含基于投票机制的影响力最大化算法。
"""

from typing import List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ...graph import IMGraph

from ..base_algorithm import BaseAlgorithm


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
        seeds: List[int] = []
        selected: Set[int] = set()
        
        for _ in range(k):
            votes = {v: 0.0 for v in range(n)}
            
            for v in range(n):
                if vote_ability[v] > 0:
                    for u, _ in self.graph.out_neighbors_with_weights(v):
                        votes[u] += vote_ability[v]
            
            for v in selected:
                votes[v] = -1
            
            max_vote = max(votes.values())
            if max_vote <= 0:
                break
            
            max_node = max(votes, key=votes.get)
            seeds.append(max_node)
            selected.add(max_node)
            
            for u, _ in self.graph.out_neighbors_with_weights(max_node):
                vote_ability[u] = max(0, vote_ability[u] - 1.0 / self.graph.out_degree(max_node))
        
        self.seeds = seeds
        return set(seeds)
