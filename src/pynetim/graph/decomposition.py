"""图分解算法模块。

提供 K-shell、K-core 等图分解算法。
"""

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import IMGraph


def compute_k_shell_values(graph: 'IMGraph') -> Dict[int, int]:
    """计算图中所有节点的 K-shell 值。

    使用 Batagelj-Zaversnik 算法，时间复杂度 O(m)。

    K-shell 分解是一种识别网络核心层次结构的方法：
    - K-shell 值高的节点位于网络核心
    - K-shell 值低的节点位于网络边缘

    Args:
        graph: IMGraph 图对象。

    Returns:
        Dict[int, int]: 节点到 K-shell 值的映射。

    References:
        Batagelj, V., & Zaversnik, M. (2003). An O(m) algorithm for 
        cores decomposition of networks. arXiv preprint cs/0310049.

        Kitsak, M., Gallos, L. K., Havlin, S., Liljeros, F., Muchnik, L., 
        Stanley, H. E., & Makse, H. A. (2010). Identification of influential 
        spreaders in complex networks. Nature physics, 6(11), 888-893.

    Example:
        >>> from pynetim import IMGraph
        >>> from pynetim.graph import compute_k_shell_values
        >>> 
        >>> graph = IMGraph(edges, directed=True)
        >>> k_shell = compute_k_shell_values(graph)
        >>> print(k_shell)  # {0: 1, 1: 2, 2: 1, ...}
    """
    n = graph.num_nodes
    if n == 0:
        return {}
    
    nodes = list(range(n))
    degree = {v: graph.out_degree(v) for v in nodes}
    
    md = max(degree.values()) if degree else 0
    
    bins = [0] * (md + 1)
    for node in nodes:
        bins[degree[node]] += 1
    
    start = 1
    for d in range(md + 1):
        num = bins[d]
        bins[d] = start
        start += num
    
    pos = {node: -1 for node in nodes}
    vert = {p: -1 for p in range(1, n + 1)}
    
    for node in nodes:
        pos[node] = bins[degree[node]]
        vert[pos[node]] = node
        bins[degree[node]] += 1
    
    for d in range(md, 0, -1):
        bins[d] = bins[d - 1]
    bins[0] = 1
    
    nbrs = {v: list(graph.out_neighbors(v)) for v in nodes}
    
    for p in range(1, n + 1):
        v = vert[p]
        for u in nbrs[v]:
            if degree[u] > degree[v]:
                du = degree[u]
                pu = pos[u]
                pw = bins[du]
                w = vert[pw]
                
                if u != w:
                    pos[u] = pw
                    vert[pu] = w
                    pos[w] = pu
                    vert[pw] = u
                
                bins[du] += 1
                degree[u] -= 1
    
    return degree


__all__ = ['compute_k_shell_values']
