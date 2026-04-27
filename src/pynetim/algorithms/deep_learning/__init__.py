"""深度学习影响力最大化算法模块。

此模块包含纯深度学习算法（非强化学习）。
深度强化学习算法已移动到 reinforcement_learning.deep 模块。
"""

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

__all__ = ['BaseDLAlgorithm']
