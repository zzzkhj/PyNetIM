try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torch_geometric
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

try:
    import torch_scatter
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False

DEEP_LEARNING_AVAILABLE = TORCH_AVAILABLE and TORCH_GEOMETRIC_AVAILABLE and TORCH_SCATTER_AVAILABLE

if not DEEP_LEARNING_AVAILABLE:
    import warnings
    missing = []
    if not TORCH_AVAILABLE:
        missing.append("torch")
    if not TORCH_GEOMETRIC_AVAILABLE:
        missing.append("torch_geometric")
    if not TORCH_SCATTER_AVAILABLE:
        missing.append("torch_scatter")
    warnings.warn(
        f"深度学习依赖未安装 ({', '.join(missing)})，深度学习算法不可用。\n"
        "请使用 'pip install pynetim[deep-learning]' 安装。",
        ImportWarning
    )

from .base_dl import BaseDLAlgorithm

if DEEP_LEARNING_AVAILABLE:
    from .touplegdd import ToupleGDDAlgorithm, S2VDQNAlgorithm, ToupleGDDTrainer, S2VDQNTrainer
    from .bigdn import BiGDNAlgorithm, BiGDNSAlgorithm, BiGDNTrainer, BiGDNNodeEncoderTrainer
    __all__ = [
        'BaseDLAlgorithm',
        'ToupleGDDAlgorithm',
        'S2VDQNAlgorithm',
        'ToupleGDDTrainer',
        'S2VDQNTrainer',
        'BiGDNAlgorithm',
        'BiGDNSAlgorithm',
        'BiGDNTrainer',
        'BiGDNNodeEncoderTrainer',
    ]
else:
    __all__ = ['BaseDLAlgorithm']
