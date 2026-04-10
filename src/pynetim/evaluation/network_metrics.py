# -*- coding: utf-8 -*-
"""网络结构评估指标。

提供网络结构特征的评估指标函数。
"""

from typing import List, Set, Dict, Optional, TYPE_CHECKING
import numpy as np
from collections import Counter

if TYPE_CHECKING:
    from ..graph import IMGraph


def distribution_entropy(
    graph: 'IMGraph',
    seeds: Set[int]
) -> float:
    """计算种子节点在度分布上的熵。
    
    熵值越高表示种子节点分布越均匀。
    
    Args:
        graph: 图对象。
        seeds: 种子节点集合。
    
    Returns:
        float: 分布熵，范围 [0, 1]。
            - 1.0 表示分布完全均匀
            - 0.0 表示分布完全集中
    
    Example:
        >>> from pynetim.evaluation import distribution_entropy
        >>> entropy = distribution_entropy(graph, seeds)
        >>> print(f"Distribution entropy: {entropy:.4f}")
    """
    if len(seeds) == 0:
        return 0.0
    
    degrees = graph.batch_out_degree(list(seeds))
    
    degree_counts = Counter(degrees)
    
    total = len(seeds)
    probabilities = [count / total for count in degree_counts.values()]
    
    entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
    
    max_entropy = np.log(len(degree_counts)) if len(degree_counts) > 1 else 1.0
    
    return entropy / max_entropy if max_entropy > 0 else 0.0


def local_clustering(
    graph: 'IMGraph',
    seeds: Set[int]
) -> float:
    """计算种子节点的局部聚类系数。
    
    评估种子节点周围网络的紧密程度。
    
    Args:
        graph: 图对象。
        seeds: 种子节点集合。
    
    Returns:
        float: 平均局部聚类系数，范围 [0, 1]。
    
    Example:
        >>> from pynetim.evaluation import local_clustering
        >>> clustering = local_clustering(graph, seeds)
        >>> print(f"Local clustering: {clustering:.4f}")
    """
    from .seed_quality_metrics import clustering_coefficient
    
    return clustering_coefficient(graph, seeds)


def reachability(
    graph: 'IMGraph',
    seeds: Set[int]
) -> float:
    """计算种子节点的可达性。
    
    可达性 = 从种子节点可达的节点数 / 网络总节点数
    
    Args:
        graph: 图对象。
        seeds: 种子节点集合。
    
    Returns:
        float: 可达性，范围 [0, 1]。
    
    Example:
        >>> from pynetim.evaluation import reachability
        >>> reach = reachability(graph, seeds)
        >>> print(f"Reachability: {reach:.2%}")
    """
    if len(seeds) == 0:
        return 0.0
    
    reachable_nodes = set(seeds)
    
    for seed in seeds:
        visited = {seed}
        queue = [seed]
        
        while queue:
            node = queue.pop(0)
            for neighbor in graph.out_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        reachable_nodes.update(visited)
    
    return len(reachable_nodes) / graph.num_nodes
