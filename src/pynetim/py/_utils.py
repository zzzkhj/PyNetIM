import random
from typing import Union
from networkx import Graph, DiGraph


def set_edge_weight(graph: Union[Graph, DiGraph], edge_weight_type: str, constant_weight: float = None):
    edge_weight_type = edge_weight_type.upper()
    
    if edge_weight_type == 'CONSTANT':
        if constant_weight is None:
            raise ValueError('使用CONSTANT模型时必须提供常量权重值')
        for u, v in graph.edges():
            graph.edges[(u, v)]['weight'] = constant_weight
    elif edge_weight_type == 'TV':
        weight_list = [0.001, 0.01, 0.1]
        for u, v in graph.edges():
            graph.edges[(u, v)]['weight'] = random.choice(weight_list)
    elif edge_weight_type == 'WC':
        for u, v in graph.edges():
            if graph.is_directed():
                graph.edges[(u, v)]['weight'] = 1 / graph.in_degree(v)
            else:
                graph.edges[(u, v)]['weight'] = 1 / graph.degree(v)
    else:
        raise ValueError('不支持的边权重模型')


def infection_threshold(graph: Union[Graph, DiGraph]) -> float:
    degree_dict = dict(graph.degree())
    degrees = list(degree_dict.values())
    k = sum(degrees)
    k2 = sum(x ** 2 for x in degrees)
    beta = k / (k2 - k)
    return beta + 0.05 * beta
