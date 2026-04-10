# -*- coding: utf-8 -*-
"""种子节点质量评估指标。

提供种子节点集合质量的评估指标函数。
"""

from typing import List, Set, Dict, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..graph import IMGraph


def neighbor_coverage(
    graph: 'IMGraph',
    seeds: Set[int]
) -> float:
    """计算种子节点的邻居覆盖率。
    
    邻居覆盖率 = 种子节点的唯一邻居数 / 网络总节点数
    
    Args:
        graph: 图对象。
        seeds: 种子节点集合。
    
    Returns:
        float: 邻居覆盖率，范围 [0, 1]。
    
    Example:
        >>> from pynetim.evaluation import neighbor_coverage
        >>> coverage = neighbor_coverage(graph, seeds)
        >>> print(f"Neighbor coverage: {coverage:.2%}")
    """
    if len(seeds) == 0:
        return 0.0
    
    neighbors = set()
    for seed in seeds:
        neighbors.update(graph.out_neighbors(seed))
    
    neighbors -= seeds
    
    return len(neighbors) / graph.num_nodes


def degree_statistics(
    graph: 'IMGraph',
    seeds: Set[int]
) -> Dict[str, float]:
    """计算种子节点的度统计信息。
    
    Args:
        graph: 图对象。
        seeds: 种子节点集合。
    
    Returns:
        Dict[str, float]: 包含以下统计量：
            - mean_degree: 平均度
            - max_degree: 最大度
            - min_degree: 最小度
            - std_degree: 度标准差
    
    Example:
        >>> from pynetim.evaluation import degree_statistics
        >>> stats = degree_statistics(graph, seeds)
        >>> print(f"Mean degree: {stats['mean_degree']:.2f}")
    """
    if len(seeds) == 0:
        return {
            'mean_degree': 0.0,
            'max_degree': 0,
            'min_degree': 0,
            'std_degree': 0.0
        }
    
    degrees = graph.batch_out_degree(list(seeds))
    
    return {
        'mean_degree': float(np.mean(degrees)),
        'max_degree': int(np.max(degrees)),
        'min_degree': int(np.min(degrees)),
        'std_degree': float(np.std(degrees))
    }


def degree_distribution(
    graph: 'IMGraph',
    seeds: Set[int]
) -> Dict[int, int]:
    """计算种子节点的度分布。
    
    Args:
        graph: 图对象。
        seeds: 种子节点集合。
    
    Returns:
        Dict[int, int]: 度值到节点数量的映射。
    
    Example:
        >>> from pynetim.evaluation import degree_distribution
        >>> dist = degree_distribution(graph, seeds)
        >>> print(f"Degree 5: {dist.get(5, 0)} nodes")
    """
    distribution = {}
    for seed in seeds:
        degree = graph.out_degree(seed)
        distribution[degree] = distribution.get(degree, 0) + 1
    
    return distribution


def mean_centrality(
    graph: 'IMGraph',
    seeds: Set[int],
    centrality_type: str = 'degree'
) -> float:
    """计算种子节点的平均中心性。
    
    Args:
        graph: 图对象。
        seeds: 种子节点集合。
        centrality_type: 中心性类型，可选：
            - 'degree': 度中心性
            - 'in_degree': 入度中心性
            - 'out_degree': 出度中心性
    
    Returns:
        float: 平均中心性值。
    
    Example:
        >>> from pynetim.evaluation import mean_centrality
        >>> centrality = mean_centrality(graph, seeds, centrality_type='degree')
    """
    if len(seeds) == 0:
        return 0.0
    
    n = graph.num_nodes
    
    if centrality_type == 'degree':
        centralities = [graph.out_degree(seed) / (n - 1) for seed in seeds]
    elif centrality_type == 'in_degree':
        centralities = [graph.in_degree(seed) / (n - 1) for seed in seeds]
    elif centrality_type == 'out_degree':
        centralities = [graph.out_degree(seed) / (n - 1) for seed in seeds]
    else:
        raise ValueError(f"Unknown centrality type: {centrality_type}")
    
    return float(np.mean(centralities))


def seed_overlap(
    seeds1: Set[int],
    seeds2: Set[int]
) -> float:
    """计算两组种子节点的重叠率。
    
    Jaccard相似度 = |S1 ∩ S2| / |S1 ∪ S2|
    
    Args:
        seeds1: 第一组种子节点。
        seeds2: 第二组种子节点。
    
    Returns:
        float: 重叠率，范围 [0, 1]。
    
    Example:
        >>> from pynetim.evaluation import seed_overlap
        >>> overlap = seed_overlap(seeds1, seeds2)
        >>> print(f"Overlap: {overlap:.2%}")
    """
    if len(seeds1) == 0 and len(seeds2) == 0:
        return 1.0
    
    intersection = len(seeds1 & seeds2)
    union = len(seeds1 | seeds2)
    
    return intersection / union if union > 0 else 0.0


def seed_diversity(
    graph: 'IMGraph',
    seeds: Set[int]
) -> float:
    """计算种子节点的多样性。
    
    基于种子节点之间的平均距离评估多样性。
    
    Args:
        graph: 图对象。
        seeds: 种子节点集合。
    
    Returns:
        float: 多样性得分，范围 [0, 1]。
            - 1.0 表示种子节点分布非常分散
            - 0.0 表示种子节点非常集中
    
    Example:
        >>> from pynetim.evaluation import seed_diversity
        >>> diversity = seed_diversity(graph, seeds)
    """
    from .influence_metrics import average_shortest_distance
    
    if len(seeds) < 2:
        return 0.0
    
    avg_distance = average_shortest_distance(graph, seeds)
    
    if avg_distance < 0:
        return 0.0
    
    max_possible_distance = graph.num_nodes - 1
    
    return min(avg_distance / max_possible_distance, 1.0)


def weight_statistics(
    graph: 'IMGraph',
    seeds: Set[int]
) -> Dict[str, float]:
    """计算种子节点相关边的权重统计信息。
    
    Args:
        graph: 图对象。
        seeds: 种子节点集合。
    
    Returns:
        Dict[str, float]: 包含以下统计量：
            - mean_weight: 平均权重
            - max_weight: 最大权重
            - min_weight: 最小权重
            - total_weight: 总权重
    
    Example:
        >>> from pynetim.evaluation import weight_statistics
        >>> stats = weight_statistics(graph, seeds)
        >>> print(f"Mean weight: {stats['mean_weight']:.4f}")
    """
    if len(seeds) == 0:
        return {
            'mean_weight': 0.0,
            'max_weight': 0.0,
            'min_weight': 0.0,
            'total_weight': 0.0
        }
    
    weights = []
    for seed in seeds:
        for neighbor in graph.out_neighbors(seed):
            weight = graph.get_edge_weight(seed, neighbor)
            if weight is not None:
                weights.append(weight)
    
    if len(weights) == 0:
        return {
            'mean_weight': 0.0,
            'max_weight': 0.0,
            'min_weight': 0.0,
            'total_weight': 0.0
        }
    
    return {
        'mean_weight': float(np.mean(weights)),
        'max_weight': float(np.max(weights)),
        'min_weight': float(np.min(weights)),
        'total_weight': float(np.sum(weights))
    }


def clustering_coefficient(
    graph: 'IMGraph',
    seeds: Set[int]
) -> float:
    """计算种子节点的平均聚类系数。
    
    聚类系数衡量节点邻居之间的连接密度。
    
    Args:
        graph: 图对象。
        seeds: 种子节点集合。
    
    Returns:
        float: 平均聚类系数，范围 [0, 1]。
    
    Example:
        >>> from pynetim.evaluation import clustering_coefficient
        >>> cc = clustering_coefficient(graph, seeds)
        >>> print(f"Clustering coefficient: {cc:.4f}")
    """
    if len(seeds) == 0:
        return 0.0
    
    coefficients = []
    
    for seed in seeds:
        neighbors = list(graph.out_neighbors(seed))
        k = len(neighbors)
        
        if k < 2:
            coefficients.append(0.0)
            continue
        
        actual_edges = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if graph.has_edge(neighbors[i], neighbors[j]):
                    actual_edges += 1
        
        possible_edges = k * (k - 1) / 2
        cc = actual_edges / possible_edges if possible_edges > 0 else 0.0
        coefficients.append(cc)
    
    return float(np.mean(coefficients))
