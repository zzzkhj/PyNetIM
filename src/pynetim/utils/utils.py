import random
from typing import Union, Dict, Any, List, TYPE_CHECKING
from collections import Counter

import networkx as nx
from networkx import Graph, DiGraph

if TYPE_CHECKING:
    from pynetim.cpp.graph import IMGraphCpp

def set_edge_weight(graph: Union[Graph, DiGraph, 'IMGraphCpp'], edge_weight_type: str, constant_weight: float = None):
    """
    根据指定的权重模型为图中的边设置权重值。

    参数:
        graph (Graph|DiGraph|IMGraphCpp): 网络图对象，可以是有向图、无向图或C++图
        edge_weight_type (str): 边权重类型，支持 'CONSTANT'、'TV'、'WC' 三种模式
        constant_weight (float, optional): 当使用常量权重模式时，指定的权重值

    异常:
        ValueError: 当权重类型不支持或参数不符合要求时抛出
    """
    edge_weight_type = edge_weight_type.upper()
    
    is_cpp_graph = hasattr(graph, 'num_nodes') and hasattr(graph, 'directed')
    
    if edge_weight_type == 'CONSTANT':
        if constant_weight is None:
            raise ValueError('使用CONSTANT模型时必须提供常量权重值')
        if is_cpp_graph:
            for u, v in graph.edges.keys():
                graph.update_edge_weight(u, v, constant_weight)
        else:
            for u, v in graph.edges():
                graph.edges[(u, v)]['weight'] = constant_weight
    elif edge_weight_type == 'TV':
        weight_list = [0.001, 0.01, 0.1]
        if is_cpp_graph:
            for u, v in graph.edges.keys():
                graph.update_edge_weight(u, v, random.choice(weight_list))
        else:
            for u, v in graph.edges():
                graph.edges[(u, v)]['weight'] = random.choice(weight_list)
    elif edge_weight_type == 'WC':
        if is_cpp_graph:
            for u, v in graph.edges.keys():
                if graph.directed:
                    graph.update_edge_weight(u, v, 1.0 / graph.in_degree(v))
                else:
                    graph.update_edge_weight(u, v, 1.0 / graph.degree(v))
        else:
            for u, v in graph.edges():
                if graph.is_directed():
                    graph.edges[(u, v)]['weight'] = 1 / graph.in_degree(v)
                else:
                    graph.edges[(u, v)]['weight'] = 1 / graph.degree(v)
    else:
        raise ValueError('不支持的边权重模型')


def infection_threshold(graph: Union[Graph, DiGraph, 'IMGraphCpp']) -> float:
    """
    计算基于图graph的度分布的感染阈值。

    参数:
        graph (Graph|DiGraph|IMGraphCpp): 网络图对象，可以是有向图、无向图或C++图

    返回:
        float: 感染阈值（是1.05倍的阈值）。

    说明:
        该函数首先计算图中所有节点的度之和（k），然后计算所有节点度的平方和（k2）。
        感染阈值计算公式为 k / (k2 - k)，其中k2是度的平方和，k是度的总和。
        这个阈值在某些流行病学模型（如SIR模型）中用于预测疾病传播的临界条件。

    参考文献:
        - Pastor-Satorras, R., & Vespignani, A. (2001). "Epidemic spreading in scale-free networks."
          Physical Review Letters, 86(14), 3200-3203.
          DOI: 10.1103/PhysRevLett.86.3200
          URL: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.3200

        - Pastor-Satorras, R., Castellano, C., Van Mieghem, P., & Vespignani, A. (2015). 
          "Epidemic processes in complex networks." Reviews of Modern Physics, 87(3), 925-979.
          DOI: 10.1103/RevModPhys.87.925
          URL: https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.87.925
    """
    is_cpp_graph = hasattr(graph, 'num_nodes') and hasattr(graph, 'directed')
    
    if is_cpp_graph:
        degrees = graph.get_all_degrees()
    else:
        degree_dict = dict(graph.degree())
        degrees = list(degree_dict.values())
    
    k = sum(degrees)
    k2 = sum(x ** 2 for x in degrees)
    beta = k / (k2 - k)
    return beta + 0.05 * beta


def topk(res_dict: dict, k: int, largest: bool = True) -> List:
    """
    从字典中返回具有最大（或最小）k个值的键的列表。

    参数:
        res_dict (dict): 输入的字典，其值用于比较以决定键的排序。
        k (int): 要返回的键的数量。
        largest (bool): 如果为True，则返回具有最大值的k个键；如果为False，则返回具有最小值的k个键。

    返回:
        list: 一个包含k个键的列表，这些键对应于字典中最大（或最小）的值。
    """
    sorted_keys = sorted(res_dict.keys(), key=lambda x: res_dict[x], reverse=largest)
    return sorted_keys[:k]


def truncate_padding(seq, max_len, pad=0):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad] * (max_len - len(seq))


def graph_statistics(graph: Union[Graph, DiGraph, 'IMGraphCpp']) -> Dict[str, Any]:
    """
    计算并返回图的详细统计信息（支持NetworkX图和C++图）

    参数:
        graph (Graph|DiGraph|IMGraphCpp): 网络图对象，可以是有向图、无向图或C++图

    返回:
        dict: 包含以下统计信息的字典
            - num_nodes: 节点数
            - num_edges: 边数
            - avg_degree: 平均度
            - max_degree: 最大度
            - min_degree: 最小度
            - degree_distribution: 度分布
            - density: 图密度
            - is_connected: 是否连通（仅NetworkX图支持）
            - num_components: 连通分量数（仅NetworkX图支持）
    """
    is_cpp_graph = hasattr(graph, 'num_nodes') and hasattr(graph, 'directed')
    
    if is_cpp_graph:
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        degrees = graph.get_all_degrees()
    else:
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        degrees = list(dict(graph.degree()).values())
    
    if num_nodes == 0:
        return {
            'num_nodes': 0,
            'num_edges': 0,
            'avg_degree': 0.0,
            'max_degree': 0,
            'min_degree': 0,
            'degree_distribution': {},
            'density': 0.0,
            'is_connected': None,
            'num_components': None
        }
    
    avg_degree = sum(degrees) / num_nodes
    max_degree = max(degrees)
    min_degree = min(degrees)
    degree_dist = dict(Counter(degrees))
    
    if is_cpp_graph:
        is_connected = None
        num_components = None
    else:
        is_connected = nx.is_connected(graph)
        num_components = nx.number_connected_components(graph)
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'min_degree': min_degree,
        'degree_distribution': degree_dist,
        'density': graph_density(graph),
        'is_connected': is_connected,
        'num_components': num_components
    }


def graph_density(graph: Union[Graph, DiGraph, 'IMGraphCpp']) -> float:
    """
    计算图的密度（支持NetworkX图和C++图）

    参数:
        graph (Graph|DiGraph|IMGraphCpp): 网络图对象，可以是有向图、无向图或C++图

    返回:
        float: 图密度值，范围在[0, 1]之间

    说明:
        图密度定义为实际边数与可能的最大边数的比值
        有向图: |E| / (|V| * (|V| - 1))
        无向图: 2 * |E| / (|V| * (|V| - 1))
    """
    is_cpp_graph = hasattr(graph, 'num_nodes') and hasattr(graph, 'directed')
    
    if is_cpp_graph:
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        is_directed = graph.directed
    else:
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        is_directed = graph.is_directed()
    
    if num_nodes <= 1:
        return 0.0
    
    if is_directed:
        max_possible_edges = num_nodes * (num_nodes - 1)
    else:
        max_possible_edges = num_nodes * (num_nodes - 1) / 2.0
    
    return float(num_edges) / max_possible_edges


def connectivity_analysis(graph: Union[Graph, DiGraph, 'IMGraphCpp']) -> Dict[str, Any]:
    """
    分析图的连通性（支持NetworkX图和C++图）

    参数:
        graph (Graph|DiGraph|IMGraphCpp): 网络图对象，可以是有向图、无向图或C++图

    返回:
        dict: 包含以下连通性信息的字典
            - is_connected: 是否连通（仅NetworkX图支持）
            - num_components: 连通分量数（仅NetworkX图支持）
            - largest_component_size: 最大连通分量大小（仅NetworkX图支持）
            - component_sizes: 所有连通分量大小列表（仅NetworkX图支持）
            - weakly_connected: 是否弱连通（仅NetworkX图支持）
    """
    is_cpp_graph = hasattr(graph, 'num_nodes') and hasattr(graph, 'directed')
    
    if is_cpp_graph:
        return {
            'is_connected': None,
            'num_components': None,
            'largest_component_size': None,
            'component_sizes': None,
            'weakly_connected': None
        }
    
    num_nodes = graph.number_of_nodes()
    
    if num_nodes == 0:
        return {
            'is_connected': True,
            'num_components': 0,
            'largest_component_size': 0,
            'component_sizes': [],
            'weakly_connected': True
        }
    
    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
        is_connected = nx.is_weakly_connected(graph)
    else:
        components = list(nx.connected_components(graph))
        is_connected = nx.is_connected(graph)
    
    component_sizes = [len(comp) for comp in components]
    largest_component_size = max(component_sizes) if component_sizes else 0
    
    return {
        'is_connected': is_connected,
        'num_components': len(components),
        'largest_component_size': largest_component_size,
        'component_sizes': component_sizes,
        'weakly_connected': is_connected
    }
