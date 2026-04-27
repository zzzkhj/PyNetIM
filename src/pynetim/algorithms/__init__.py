import warnings

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

from .population import BasePopulationAlgorithm, SADPEAAlgorithm
try:
    from .population import RLSetGWOAlgorithm
except ImportError:
    RLSetGWOAlgorithm = None

from .reinforcement_learning import (
    BaseRLAlgorithm, CoreQAlgorithm, TCQAlgorithm,
)

_DEEP_ALGORITHMS_AVAILABLE = True
try:
    from .reinforcement_learning import (
        BaseDRLAlgorithm,
        BiGDNAlgorithm, BiGDNSAlgorithm, BiGDNTrainer, BiGDNNodeEncoderTrainer,
        ToupleGDDAlgorithm, S2VDQNAlgorithm, ToupleGDDTrainer, S2VDQNTrainer, IMEnvironment
    )
except ImportError:
    _DEEP_ALGORITHMS_AVAILABLE = False
    BaseDRLAlgorithm = None
    BiGDNAlgorithm = None
    BiGDNSAlgorithm = None
    BiGDNTrainer = None
    BiGDNNodeEncoderTrainer = None
    ToupleGDDAlgorithm = None
    S2VDQNAlgorithm = None
    ToupleGDDTrainer = None
    S2VDQNTrainer = None
    IMEnvironment = None

try:
    from .deep_learning import BaseDLAlgorithm
except ImportError:
    BaseDLAlgorithm = None

if not _DEEP_ALGORITHMS_AVAILABLE or RLSetGWOAlgorithm is None or BaseDLAlgorithm is None:
    _missing = []
    if RLSetGWOAlgorithm is None:
        _missing.append("RLSetGWOAlgorithm")
    if not _DEEP_ALGORITHMS_AVAILABLE:
        _missing.extend([
            "BaseDRLAlgorithm", "BiGDNAlgorithm", "BiGDNSAlgorithm",
            "ToupleGDDAlgorithm", "S2VDQNAlgorithm",
        ])
    if BaseDLAlgorithm is None:
        _missing.append("BaseDLAlgorithm")
    warnings.warn(
        f"以下算法需要 torch 等依赖但不可用: {', '.join(_missing)}\n"
        "请使用 'pip install pynetim[deep-learning]' 安装。",
        ImportWarning
    )

__all__ = [
    'BaseAlgorithm',
    'BasePopulationAlgorithm',
    'BaseRLAlgorithm',
    'BaseDRLAlgorithm',
    'CoreQAlgorithm',
    'TCQAlgorithm',
    'RLSetGWOAlgorithm',
    'SADPEAAlgorithm',
    'BiGDNAlgorithm',
    'BiGDNSAlgorithm',
    'BiGDNTrainer',
    'BiGDNNodeEncoderTrainer',
    'ToupleGDDAlgorithm',
    'S2VDQNAlgorithm',
    'ToupleGDDTrainer',
    'S2VDQNTrainer',
    'IMEnvironment',
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
