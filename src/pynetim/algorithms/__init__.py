from .base_algorithm import BaseAlgorithm
from .heuristic import (
    DegreeCentralityAlgorithm,
    PageRankAlgorithm,
    VoteRankAlgorithm,
    KShellDecompositionAlgorithm,
    BetweennessCentralityAlgorithm,
    ClosenessCentralityAlgorithm,
    EigenvectorCentralityAlgorithm,
    SingleDiscountAlgorithm,
    DegreeDiscountAlgorithm,
)
from .simulation import GreedyAlgorithm, CELFAlgorithm, CELFPlusAlgorithm
from .ris import BaseRISAlgorithm, IMMAlgorithm, TIMAlgorithm, TIMPlusAlgorithm, OPIMAlgorithm, OPIMCAlgorithm

__all__ = [
    'BaseAlgorithm',
    'DegreeCentralityAlgorithm',
    'PageRankAlgorithm',
    'VoteRankAlgorithm',
    'KShellDecompositionAlgorithm',
    'BetweennessCentralityAlgorithm',
    'ClosenessCentralityAlgorithm',
    'EigenvectorCentralityAlgorithm',
    'SingleDiscountAlgorithm',
    'DegreeDiscountAlgorithm',
    'GreedyAlgorithm',
    'CELFAlgorithm',
    'CELFPlusAlgorithm',
    'BaseRISAlgorithm',
    'IMMAlgorithm',
    'TIMAlgorithm',
    'TIMPlusAlgorithm',
    'OPIMAlgorithm',
    'OPIMCAlgorithm',
]

try:
    from .deep_learning import ToupleGDDAlgorithm, S2VDQNAlgorithm, Trainer, IMEnvironment
    __all__.extend(['ToupleGDDAlgorithm', 'S2VDQNAlgorithm', 'Trainer', 'IMEnvironment'])
except ImportError:
    pass
