# src/pynetim/__init__.py
__version__ = "0.3.0"
__author__ = "Zhang Kaijing"

# 1. 导入子包（让用户能访问 im.py 和 im.cpp）
from . import py
from . import cpp

__all__ = [
    'py',      # 导出子包本身
    'cpp',     # 导出子包本身
]