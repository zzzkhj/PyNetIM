# -*- coding: utf-8 -*-
"""排名相关评估指标。

提供种子节点排名的稳定性和单调性评估指标函数。
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np

try:
    from scipy.stats import kendalltau, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def kendall_tau(
    ranking1: Union[List[int], np.ndarray],
    ranking2: Union[List[int], np.ndarray],
    variant: str = 'b'
) -> Tuple[float, float]:
    """计算两个排名之间的肯德尔系数。
    
    Kendall's Tau 用于衡量两个排名序列的相关性，
    适用于评估不同算法得到的种子节点排名的一致性。
    
    Args:
        ranking1: 第一个排名序列（节点ID列表）。
        ranking2: 第二个排名序列（节点ID列表）。
        variant: Kendall's Tau 变体，可选 'b' 或 'c'。
                默认 'b'，可处理并列排名。
    
    Returns:
        Tuple[float, float]: (tau系数, p值)
            - tau系数范围 [-1, 1]，1表示完全一致，-1表示完全相反
            - p值表示统计显著性
    
    Raises:
        ImportError: 如果 scipy 未安装。
        ValueError: 如果输入长度不一致。
    
    Example:
        >>> from pynetim.evaluation import kendall_tau
        >>> seeds_algo1 = [0, 1, 2, 3, 4]
        >>> seeds_algo2 = [0, 2, 1, 3, 4]
        >>> tau, p = kendall_tau(seeds_algo1, seeds_algo2)
        >>> print(f"Kendall's Tau: {tau:.4f}, p-value: {p:.4f}")
    
    References:
        Kendall, M. G. (1938). A new measure of rank correlation.
        Biometrika, 30(1/2), 81-93.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for kendall_tau. "
            "Install it with: pip install scipy"
        )
    
    ranking1 = np.array(ranking1)
    ranking2 = np.array(ranking2)
    
    if len(ranking1) != len(ranking2):
        raise ValueError(
            f"Rankings must have the same length. "
            f"Got {len(ranking1)} and {len(ranking2)}."
        )
    
    if len(ranking1) == 0:
        raise ValueError("Rankings cannot be empty.")
    
    result = kendalltau(ranking1, ranking2, variant=variant)
    return result.correlation, result.pvalue


def spearman_correlation(
    ranking1: Union[List[int], np.ndarray],
    ranking2: Union[List[int], np.ndarray]
) -> Tuple[float, float]:
    """计算两个排名之间的斯皮尔曼相关系数。
    
    Spearman相关系数评估两个排名的单调关系，
    适用于评估种子节点排名的相关性。
    
    Args:
        ranking1: 第一个排名序列。
        ranking2: 第二个排名序列。
    
    Returns:
        Tuple[float, float]: (相关系数, p值)
            - 相关系数范围 [-1, 1]
            - p值表示统计显著性
    
    Raises:
        ImportError: 如果 scipy 未安装。
        ValueError: 如果输入长度不一致。
    
    Example:
        >>> from pynetim.evaluation import spearman_correlation
        >>> ranking1 = [1, 2, 3, 4, 5]
        >>> ranking2 = [1, 3, 2, 4, 5]
        >>> rho, p = spearman_correlation(ranking1, ranking2)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for spearman_correlation. "
            "Install it with: pip install scipy"
        )
    
    ranking1 = np.array(ranking1)
    ranking2 = np.array(ranking2)
    
    if len(ranking1) != len(ranking2):
        raise ValueError(
            f"Rankings must have the same length. "
            f"Got {len(ranking1)} and {len(ranking2)}."
        )
    
    if len(ranking1) == 0:
        raise ValueError("Rankings cannot be empty.")
    
    result = spearmanr(ranking1, ranking2)
    return result.correlation, result.pvalue


def monotonicity_score(
    values: Union[List[float], np.ndarray]
) -> float:
    """计算节点重要性排序的单调性得分。
    
    单调性指标衡量排序算法能否有效区分不同节点的重要性。
    如果算法能给每个节点赋予唯一的重要性值，单调性接近1；
    如果很多节点有相同的重要性值，单调性会降低。
    
    计算公式：
        M = (N_unique - 1) / (N_total - 1)
    
    其中：
        - N_unique: 不同重要性值的数量
        - N_total: 总节点数
        - M 的取值范围为 [0, 1]
        - M → 1 表示排序算法具有良好的区分度
        - M → 0 表示所有节点的重要性值都相同
    
    Args:
        values: 节点重要性得分序列。
    
    Returns:
        float: 单调性得分，范围 [0, 1]。
            - 1.0 表示所有节点的重要性值都唯一（完美区分）
            - 0.0 表示所有节点的重要性值都相同（无区分度）
    
    Example:
        >>> from pynetim.evaluation import monotonicity_score
        >>> # 所有值都不同，单调性高
        >>> values1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> score1 = monotonicity_score(values1)
        >>> print(f"Monotonicity: {score1:.4f}")  # 1.0
        
        >>> # 有重复值，单调性降低
        >>> values2 = [1.0, 2.0, 2.0, 4.0, 5.0]
        >>> score2 = monotonicity_score(values2)
        >>> print(f"Monotonicity: {score2:.4f}")  # 0.75
        
        >>> # 所有值相同，单调性为0
        >>> values3 = [3.0, 3.0, 3.0, 3.0, 3.0]
        >>> score3 = monotonicity_score(values3)
        >>> print(f"Monotonicity: {score3:.4f}")  # 0.0
    
    References:
        - A novel voting measure for identifying influential nodes in complex 
          networks based on local structure. Scientific Reports, 2025.
        - 复杂网络节点重要性排序算法的单调性评估。
    """
    values = np.array(values)
    
    if len(values) < 2:
        return 1.0
    
    unique_values = np.unique(values)
    n_unique = len(unique_values)
    n_total = len(values)
    
    if n_unique == 1:
        return 0.0
    
    monotonicity = (n_unique - 1) / (n_total - 1)
    
    return float(monotonicity)


def ranking_distance(
    ranking1: Union[List[int], np.ndarray],
    ranking2: Union[List[int], np.ndarray],
    metric: str = 'kendall'
) -> float:
    """计算两个排名之间的距离。
    
    Args:
        ranking1: 第一个排名序列。
        ranking2: 第二个排名序列。
        metric: 距离度量方法，可选：
            - 'kendall': Kendall距离（交换次数）
            - 'spearman': Spearman距离（排名差的平方和）
            - 'hamming': Hamming距离（不同位置的个数）
    
    Returns:
        float: 排名距离，越小表示排名越相似。
    
    Raises:
        ValueError: 如果 metric 不支持。
    
    Example:
        >>> from pynetim.evaluation import ranking_distance
        >>> ranking1 = [0, 1, 2, 3, 4]
        >>> ranking2 = [0, 2, 1, 3, 4]
        >>> distance = ranking_distance(ranking1, ranking2, metric='kendall')
    """
    ranking1 = np.array(ranking1)
    ranking2 = np.array(ranking2)
    
    if len(ranking1) != len(ranking2):
        raise ValueError("Rankings must have the same length.")
    
    n = len(ranking1)
    
    if metric == 'kendall':
        n_swaps = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (ranking1[i] < ranking1[j]) != (ranking2[i] < ranking2[j]):
                    n_swaps += 1
        max_swaps = n * (n - 1) / 2
        return n_swaps / max_swaps if max_swaps > 0 else 0.0
    
    elif metric == 'spearman':
        rank1 = np.argsort(np.argsort(ranking1))
        rank2 = np.argsort(np.argsort(ranking2))
        return np.sum((rank1 - rank2) ** 2)
    
    elif metric == 'hamming':
        return np.sum(ranking1 != ranking2) / n
    
    else:
        raise ValueError(
            f"Unknown metric: {metric}. "
            f"Supported metrics: 'kendall', 'spearman', 'hamming'."
        )


def ranking_stability(
    rankings: List[Union[List[int], np.ndarray]],
    method: str = 'kendall'
) -> float:
    """计算多个排名之间的平均稳定性。
    
    评估算法多次运行得到的排名的一致性。
    
    Args:
        rankings: 多个排名序列的列表。
        method: 稳定性计算方法，可选：
            - 'kendall': 使用肯德尔系数
            - 'spearman': 使用斯皮尔曼系数
            - 'overlap': 使用 Top-K 重叠率
    
    Returns:
        float: 平均稳定性得分，范围 [0, 1]。
    
    Example:
        >>> from pynetim.evaluation import ranking_stability
        >>> rankings = [
        ...     [0, 1, 2, 3, 4],
        ...     [0, 2, 1, 3, 4],
        ...     [0, 1, 2, 4, 3]
        ... ]
        >>> stability = ranking_stability(rankings)
    """
    if len(rankings) < 2:
        return 1.0
    
    scores = []
    n = len(rankings)
    
    for i in range(n):
        for j in range(i + 1, n):
            if method == 'kendall':
                tau, _ = kendall_tau(rankings[i], rankings[j])
                scores.append((tau + 1) / 2)
            elif method == 'spearman':
                rho, _ = spearman_correlation(rankings[i], rankings[j])
                scores.append((rho + 1) / 2)
            elif method == 'overlap':
                from .influence_metrics import top_k_overlap
                k = min(len(rankings[i]), len(rankings[j]))
                scores.append(top_k_overlap(rankings[i], rankings[j], k))
            else:
                raise ValueError(f"Unknown method: {method}")
    
    return np.mean(scores)
