# -*- coding: utf-8 -*-
"""影响力评估指标。

提供影响力传播效果的评估指标函数。
"""

from typing import List, Set, Dict, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..graph import IMGraph


def average_shortest_distance(
    graph: 'IMGraph',
    seeds: Set[int],
    use_weight: bool = False
) -> float:
    """计算种子节点之间的平均最短距离。
    
    Args:
        graph: 图对象。
        seeds: 种子节点集合。
        use_weight: 是否使用边权重计算距离，默认 False（使用跳数）。
    
    Returns:
        float: 平均最短距离。如果种子节点之间不可达，返回 -1。
    
    Example:
        >>> from pynetim.evaluation import average_shortest_distance
        >>> # 基于跳数
        >>> avg_dist = average_shortest_distance(graph, seeds)
        >>> # 基于权重
        >>> avg_dist = average_shortest_distance(graph, seeds, use_weight=True)
    """
    from ..utils import shortest_path_length
    
    if len(seeds) < 2:
        return 0.0
    
    seeds_list = list(seeds)
    distances = []
    
    for i in range(len(seeds_list)):
        for j in range(i + 1, len(seeds_list)):
            dist = shortest_path_length(graph, seeds_list[i], seeds_list[j], use_weight)
            if dist is not None:
                distances.append(dist)
    
    if len(distances) == 0:
        return -1.0
    
    return float(np.mean(distances))


def top_k_accuracy(
    predicted_seeds: Union[Set[int], List[int]],
    ground_truth_seeds: Union[Set[int], List[int]],
    k: Optional[int] = None
) -> float:
    """计算 Top-K 准确率。
    
    评估预测的种子节点与真实种子节点的重叠程度。
    
    Args:
        predicted_seeds: 预测的种子节点（按重要性排序）。
        ground_truth_seeds: 真实的种子节点。
        k: Top-K 的 K 值，默认为 min(len(predicted), len(ground_truth))。
    
    Returns:
        float: Top-K 准确率，范围 [0, 1]。
            - 1.0 表示完全重叠
            - 0.0 表示完全不重叠
    
    Example:
        >>> from pynetim.evaluation import top_k_accuracy
        >>> predicted = [0, 1, 2, 3, 4]
        >>> ground_truth = [0, 2, 1, 5, 6]
        >>> acc = top_k_accuracy(predicted, ground_truth, k=3)
        >>> print(f"Top-3 accuracy: {acc:.2%}")
    """
    predicted = list(predicted_seeds)
    ground_truth = set(ground_truth_seeds)
    
    if k is None:
        k = min(len(predicted), len(ground_truth))
    
    k = min(k, len(predicted), len(ground_truth))
    
    if k == 0:
        return 0.0
    
    top_k_predicted = set(predicted[:k])
    
    intersection = len(top_k_predicted & ground_truth)
    
    return intersection / k


def top_k_overlap(
    ranking1: Union[List[int], Set[int]],
    ranking2: Union[List[int], Set[int]],
    k: int
) -> float:
    """计算两个排名在 Top-K 位置的重叠率。
    
    评估不同算法选择的种子节点集合的重叠程度。
    
    Args:
        ranking1: 第一个排名序列（节点ID列表）。
        ranking2: 第二个排名序列（节点ID列表）。
        k: Top-K 的 K 值。
    
    Returns:
        float: 重叠率，范围 [0, 1]。
            - 1.0 表示完全重叠
            - 0.0 表示完全不重叠
    
    Example:
        >>> from pynetim.evaluation import top_k_overlap
        >>> ranking1 = [0, 1, 2, 3, 4, 5]
        >>> ranking2 = [0, 2, 1, 4, 3, 5]
        >>> overlap = top_k_overlap(ranking1, ranking2, k=3)
        >>> print(f"Top-3 overlap: {overlap:.4f}")
    """
    ranking1 = list(ranking1)
    ranking2 = list(ranking2)
    
    if k <= 0:
        raise ValueError("k must be positive.")
    
    k = min(k, len(ranking1), len(ranking2))
    
    top_k_set1 = set(ranking1[:k])
    top_k_set2 = set(ranking2[:k])
    
    intersection = len(top_k_set1 & top_k_set2)
    
    return intersection / k
