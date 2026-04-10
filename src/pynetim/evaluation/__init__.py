# -*- coding: utf-8 -*-
"""评估指标模块。

提供影响力最大化算法的多种评估指标函数。

模块包含:
- ranking_metrics: 排名稳定性评估指标
- influence_metrics: 影响力传播评估指标
- seed_quality_metrics: 种子节点质量评估指标
- network_metrics: 网络结构评估指标
"""

from .ranking_metrics import (
    kendall_tau,
    spearman_correlation,
    monotonicity_score,
    ranking_distance,
    ranking_stability,
)

from .influence_metrics import (
    average_shortest_distance,
    top_k_accuracy,
    top_k_overlap,
)

from .seed_quality_metrics import (
    neighbor_coverage,
    degree_statistics,
    degree_distribution,
    mean_centrality,
    seed_overlap,
    seed_diversity,
    weight_statistics,
    clustering_coefficient,
)

from .network_metrics import (
    distribution_entropy,
    local_clustering,
    reachability,
)

__all__ = [
    # Ranking metrics
    'kendall_tau',
    'spearman_correlation',
    'monotonicity_score',
    'ranking_distance',
    'ranking_stability',
    
    # Influence metrics
    'average_shortest_distance',
    'top_k_accuracy',
    'top_k_overlap',
    
    # Seed quality metrics
    'neighbor_coverage',
    'degree_statistics',
    'degree_distribution',
    'mean_centrality',
    'seed_overlap',
    'seed_diversity',
    'weight_statistics',
    'clustering_coefficient',
    
    # Network metrics
    'distribution_entropy',
    'local_clustering',
    'reachability',
]
