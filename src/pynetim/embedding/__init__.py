"""图嵌入算法模块。

包含多种图节点嵌入算法，如 Node2Vec、DeepWalk、Struc2Vec、Inf2Vec 等。
"""

from .base import BaseEmbedding
from .node2vec import Node2Vec
from .deepwalk import DeepWalk
from .struc2vec import Struc2Vec
from .inf2vec import Inf2Vec

__all__ = [
    'BaseEmbedding',
    'Node2Vec',
    'DeepWalk',
    'Struc2Vec',
    'Inf2Vec',
]
