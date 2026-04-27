# -*- coding: utf-8 -*-
"""权重管理模块。

提供类似 Hugging Face Transformers 的预训练权重管理功能。
"""

from .weight_manager import WeightManager, WEIGHTS_CONFIG

__all__ = ['WeightManager', 'WEIGHTS_CONFIG']
