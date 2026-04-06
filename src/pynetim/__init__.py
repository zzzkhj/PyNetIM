# -*- coding: utf-8 -*-
__version__ = "0.5.0"
__author__ = "Zhang Kaijing"

from . import graph
from . import diffusion_model
from . import utils
from . import algorithms

from .graph import IMGraph
from .diffusion_model import (
    IndependentCascadeModel,
    LinearThresholdModel,
    SusceptibleInfectedModel,
    SusceptibleInfectedRecoveredModel,
    BaseCallbackDiffusionModel,
    BaseMultiprocessDiffusionModel,
)
from .utils import (
    renumber_edges, 
    to_networkx, 
    to_igraph, 
    to_scipy_sparse, 
    to_pyg,
    load_edgelist,
    save_edgelist
)
from .algorithms import (
    BaseAlgorithm,
    SingleDiscountAlgorithm,
    DegreeDiscountAlgorithm,
    GreedyAlgorithm,
    CELFAlgorithm,
    BaseRISAlgorithm,
    IMMAlgorithm,
)

__all__ = [
    'graph',
    'diffusion_model',
    'utils',
    'algorithms',
    'IMGraph',
    'IndependentCascadeModel',
    'LinearThresholdModel',
    'SusceptibleInfectedModel',
    'SusceptibleInfectedRecoveredModel',
    'BaseCallbackDiffusionModel',
    'BaseMultiprocessDiffusionModel',
    'renumber_edges',
    'to_networkx',
    'to_igraph',
    'to_scipy_sparse',
    'to_pyg',
    'load_edgelist',
    'save_edgelist',
    'BaseAlgorithm',
    'SingleDiscountAlgorithm',
    'DegreeDiscountAlgorithm',
    'GreedyAlgorithm',
    'CELFAlgorithm',
    'BaseRISAlgorithm',
    'IMMAlgorithm',
]
