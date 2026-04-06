import warnings

warnings.warn(
    "\n"
    "┌─────────────────────────────────────────────────────────────────┐\n"
    "│  pynetim.py 模块提醒                                            │\n"
    "├─────────────────────────────────────────────────────────────────┤\n"
    "│  该模块目前处于维护模式，不再添加新功能。                        │\n"
    "│  新功能和性能优化将集中在 C++ 主模块 (pynetim) 中。             │\n"
    "│                                                                 │\n"
    "│  推荐使用:                                                      │\n"
    "│    from pynetim import IMGraph, IndependentCascadeModel        │\n"
    "│    from pynetim.diffusion_model import BaseCallbackDiffusionModel │\n"
    "└─────────────────────────────────────────────────────────────────┘",
    UserWarning,
    stacklevel=2
)

from . import algorithms
from . import diffusion_model
from . import graph

__all__ = [
    "algorithms",
    "diffusion_model",
    "graph",
]
