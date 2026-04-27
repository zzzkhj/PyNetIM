"""强化学习影响力最大化算法模块。"""

from .base_rl import BaseRLAlgorithm
from .coreq import CoreQAlgorithm
from .tcq import TCQAlgorithm

__all__ = [
    'BaseRLAlgorithm',
    'CoreQAlgorithm',
    'TCQAlgorithm',
]

try:
    from .deep import (
        BaseDRLAlgorithm,
        BiGDNAlgorithm, BiGDNSAlgorithm, BiGDNTrainer, BiGDNNodeEncoderTrainer,
        ToupleGDDAlgorithm, S2VDQNAlgorithm, ToupleGDDTrainer, S2VDQNTrainer, IMEnvironment
    )
    __all__.extend([
        'BaseDRLAlgorithm',
        'BiGDNAlgorithm', 'BiGDNSAlgorithm', 'BiGDNTrainer', 'BiGDNNodeEncoderTrainer',
        'ToupleGDDAlgorithm', 'S2VDQNAlgorithm', 'ToupleGDDTrainer', 'S2VDQNTrainer', 'IMEnvironment'
    ])
except ImportError:
    pass
