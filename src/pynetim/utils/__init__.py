# -*- coding: utf-8 -*-
"""工具函数模块。

提供通用的工具函数。
"""

from typing import Optional, List, Tuple, TYPE_CHECKING
from collections import deque
import heapq

if TYPE_CHECKING:
    from ..graph import IMGraph

from .graph_utils import (
    load_edgelist,
    save_edgelist,
    to_networkx,
    to_igraph,
    to_scipy_sparse,
    to_pyg,
)


def renumber_edges(edges: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], dict, List[int]]:
    """重新编号边列表中的节点为连续整数。
    
    Args:
        edges: 边列表，每个元素为 (u, v) 元组。
    
    Returns:
        Tuple[List[Tuple[int, int]], dict, List[int]]: 
            - 重编号后的边列表
            - 原始节点ID到新节点ID的映射
            - 新节点ID到原始节点ID的映射
    
    Example:
        >>> from pynetim.utils import renumber_edges
        >>> edges = [(10, 20), (20, 30), (30, 40)]
        >>> new_edges, mapping, reverse_mapping = renumber_edges(edges)
        >>> print(new_edges)  # [(0, 1), (1, 2), (2, 3)]
    """
    all_nodes = set()
    for u, v in edges:
        all_nodes.add(u)
        all_nodes.add(v)
    
    sorted_nodes = sorted(all_nodes)
    
    node_to_id = {node: i for i, node in enumerate(sorted_nodes)}
    id_to_node = sorted_nodes
    
    new_edges = [(node_to_id[u], node_to_id[v]) for u, v in edges]
    
    return new_edges, node_to_id, id_to_node


def shortest_path_length(
    graph: 'IMGraph',
    source: int,
    target: int,
    use_weight: bool = False
) -> Optional[float]:
    """计算两个节点之间的最短路径长度。
    
    Args:
        graph: 图对象。
        source: 源节点。
        target: 目标节点。
        use_weight: 是否使用边权重计算最短路径。
            - False (默认): 使用跳数（边数）作为距离
            - True: 使用边权重之和作为距离
    
    Returns:
        Optional[float]: 最短路径长度，如果不可达则返回 None。
            - use_weight=False 时返回整数（跳数）
            - use_weight=True 时返回浮点数（权重和）
    
    Example:
        >>> from pynetim.utils import shortest_path_length
        >>> # 基于跳数
        >>> dist = shortest_path_length(graph, 0, 5, use_weight=False)
        >>> # 基于权重
        >>> dist = shortest_path_length(graph, 0, 5, use_weight=True)
    """
    if source == target:
        return 0.0 if use_weight else 0
    
    if use_weight:
        return _dijkstra_shortest_path(graph, source, target)
    else:
        return _bfs_shortest_path(graph, source, target)


def _bfs_shortest_path(
    graph: 'IMGraph',
    source: int,
    target: int
) -> Optional[int]:
    """BFS 计算无权最短路径（跳数）。
    
    Args:
        graph: 图对象。
        source: 源节点。
        target: 目标节点。
    
    Returns:
        Optional[int]: 最短路径跳数，如果不可达则返回 None。
    """
    visited = {source}
    queue = deque([(source, 0)])
    
    while queue:
        node, dist = queue.popleft()
        
        for neighbor in graph.out_neighbors(node):
            if neighbor == target:
                return dist + 1
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    return None


def _dijkstra_shortest_path(
    graph: 'IMGraph',
    source: int,
    target: int
) -> Optional[float]:
    """Dijkstra 算法计算带权最短路径。
    
    Args:
        graph: 图对象。
        source: 源节点。
        target: 目标节点。
    
    Returns:
        Optional[float]: 最短路径权重和，如果不可达则返回 None。
    """
    distances = {source: 0.0}
    heap = [(0.0, source)]
    visited = set()
    
    while heap:
        current_dist, node = heapq.heappop(heap)
        
        if node in visited:
            continue
        
        visited.add(node)
        
        if node == target:
            return current_dist
        
        neighbors, weights = graph.out_neighbors_arrays(node)
        
        for neighbor, weight in zip(neighbors, weights):
            if neighbor in visited:
                continue
            
            new_dist = current_dist + weight
            
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    
    return None


def all_pairs_shortest_path_length(
    graph: 'IMGraph',
    nodes: Optional[List[int]] = None,
    use_weight: bool = False
) -> dict:
    """计算节点对之间的最短路径长度。
    
    Args:
        graph: 图对象。
        nodes: 要计算的节点列表，如果为 None 则计算所有节点。
        use_weight: 是否使用边权重。
    
    Returns:
        dict: 嵌套字典，distances[u][v] 表示 u 到 v 的最短路径长度。
    
    Example:
        >>> from pynetim.utils import all_pairs_shortest_path_length
        >>> distances = all_pairs_shortest_path_length(graph, nodes=[0, 1, 2])
        >>> print(distances[0][1])
    """
    if nodes is None:
        nodes = list(range(graph.num_nodes))
    
    distances = {}
    
    for source in nodes:
        distances[source] = {}
        for target in nodes:
            if source == target:
                distances[source][target] = 0.0 if use_weight else 0
            else:
                dist = shortest_path_length(graph, source, target, use_weight)
                distances[source][target] = dist
    
    return distances


__all__ = [
    'load_edgelist',
    'save_edgelist',
    'to_networkx',
    'to_igraph',
    'to_scipy_sparse',
    'to_pyg',
    'renumber_edges',
    'shortest_path_length',
    'all_pairs_shortest_path_length',
]
