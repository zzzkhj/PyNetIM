"""启发式影响力最大化算法模块。

包含多种基于中心性和折扣的启发式算法。
"""

from .centrality import (
    DegreeCentralityAlgorithm,
    PageRankAlgorithm,
    KShellDecompositionAlgorithm,
    BetweennessCentralityAlgorithm,
    ClosenessCentralityAlgorithm,
    EigenvectorCentralityAlgorithm,
)

from .discount import (
    SingleDiscountAlgorithm,
    DegreeDiscountAlgorithm,
)

from .vote import (
    VoteRankAlgorithm,
)

try:
    from .gatsh import GATSHAlgorithm
except ImportError:
    GATSHAlgorithm = None

__all__ = [
    'DegreeCentralityAlgorithm',
    'PageRankAlgorithm',
    'VoteRankAlgorithm',
    'KShellDecompositionAlgorithm',
    'BetweennessCentralityAlgorithm',
    'ClosenessCentralityAlgorithm',
    'EigenvectorCentralityAlgorithm',
    'SingleDiscountAlgorithm',
    'DegreeDiscountAlgorithm',
    'GATSHAlgorithm',
]
