"""种群优化算法模块。

提供基于种群进化的影响力最大化算法。
"""

from .base_population import BasePopulationAlgorithm
from .sadpea import SADPEAAlgorithm

try:
    from .rlsetgwo import RLSetGWOAlgorithm
except ImportError:
    RLSetGWOAlgorithm = None

__all__ = [
    'BasePopulationAlgorithm',
    'RLSetGWOAlgorithm',
    'SADPEAAlgorithm',
]
