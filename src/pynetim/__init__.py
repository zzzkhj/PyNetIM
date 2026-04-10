# -*- coding: utf-8 -*-
__version__ = "0.5.0"
__author__ = "Zhang Kaijing"

from . import graph
from . import diffusion_model
from . import utils
from . import algorithms
from . import evaluation
from . import timing

from .graph import IMGraph
from .diffusion_model import (
    IndependentCascadeModel,
    LinearThresholdModel,
    SusceptibleInfectedModel,
    SusceptibleInfectedRecoveredModel,
    BaseCallbackDiffusionModel,
    BaseMultiprocessDiffusionModel,
)
from .utils import (
    renumber_edges, 
    to_networkx, 
    to_igraph, 
    to_scipy_sparse, 
    to_pyg,
    load_edgelist,
    save_edgelist,
    shortest_path_length,
    all_pairs_shortest_path_length,
)
from .algorithms import (
    BaseAlgorithm,
    SingleDiscountAlgorithm,
    DegreeDiscountAlgorithm,
    GreedyAlgorithm,
    CELFAlgorithm,
    BaseRISAlgorithm,
    IMMAlgorithm,
    TIMAlgorithm,
    TIMPlusAlgorithm,
    OPIMAlgorithm,
    OPIMCAlgorithm,
)
from .evaluation import (
    kendall_tau,
    spearman_correlation,
    monotonicity_score,
    ranking_distance,
    ranking_stability,
    average_shortest_distance,
    top_k_accuracy,
    top_k_overlap,
    neighbor_coverage,
    degree_statistics,
    degree_distribution,
    mean_centrality,
    seed_overlap,
    seed_diversity,
    weight_statistics,
    clustering_coefficient,
    distribution_entropy,
    local_clustering,
    reachability,
)

from .timing import (
    measure_time,
    AlgorithmTimer,
    measure_runtime,
    measure_runtime_multiple_runs,
    compare_algorithms_runtime,
)

__all__ = [
    'graph',
    'diffusion_model',
    'utils',
    'algorithms',
    'evaluation',
    'timing',
    'IMGraph',
    'IndependentCascadeModel',
    'LinearThresholdModel',
    'SusceptibleInfectedModel',
    'SusceptibleInfectedRecoveredModel',
    'BaseCallbackDiffusionModel',
    'BaseMultiprocessDiffusionModel',
    'renumber_edges',
    'to_networkx',
    'to_igraph',
    'to_scipy_sparse',
    'to_pyg',
    'load_edgelist',
    'save_edgelist',
    'shortest_path_length',
    'all_pairs_shortest_path_length',
    'BaseAlgorithm',
    'SingleDiscountAlgorithm',
    'DegreeDiscountAlgorithm',
    'GreedyAlgorithm',
    'CELFAlgorithm',
    'BaseRISAlgorithm',
    'IMMAlgorithm',
    'TIMAlgorithm',
    'TIMPlusAlgorithm',
    'OPIMAlgorithm',
    'OPIMCAlgorithm',
    'kendall_tau',
    'spearman_correlation',
    'monotonicity_score',
    'ranking_distance',
    'ranking_stability',
    'average_shortest_distance',
    'top_k_accuracy',
    'top_k_overlap',
    'neighbor_coverage',
    'degree_statistics',
    'degree_distribution',
    'mean_centrality',
    'seed_overlap',
    'seed_diversity',
    'weight_statistics',
    'clustering_coefficient',
    'distribution_entropy',
    'local_clustering',
    'reachability',
    'measure_time',
    'AlgorithmTimer',
    'measure_runtime',
    'measure_runtime_multiple_runs',
    'compare_algorithms_runtime',
]
