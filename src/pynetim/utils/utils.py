import random
from typing import Union

from networkx import Graph, DiGraph

from ..cpp.graph import IMGraphCpp

def set_edge_weight(graph: Union[Graph, DiGraph, IMGraphCpp], edge_weight_type: str, constant_weight: float = None):
    """
    根据指定的权重模型为图中的边设置权重值。

    参数:
        graph (Graph|DiGraph|IMGraph): 网络图对象，可以是有向图或无向图
        edge_weight_type (str): 边权重类型，支持 'CONSTANT'、'TV'、'WC' 三种模式
        constant_weight (float, optional): 当使用常量权重模式时，指定的权重值

    异常:
        ValueError: 当权重类型不支持或参数不符合要求时抛出
    """
    # 根据不同的权重模型设置权重
    edge_weight_type = edge_weight_type.upper()
    if edge_weight_type == 'CONSTANT':
        # 如果模型是'CONSTANT'且未提供constant_weight，则抛出错误
        if constant_weight is None:
            raise ValueError('使用CONSTANT模型时必须提供常量权重值')
        # 为每条边设置常量权重
        for u, v in graph.edges():
            if isinstance(graph, IMGraphCpp):
                graph.edges[(u, v)] = constant_weight
            else:
                graph.edges[(u, v)]['weight'] = constant_weight
    elif edge_weight_type == 'TV':
        # 定义权重列表，用于随机选择
        weight_list = [0.001, 0.01, 0.1]
        # 为每条边随机选择一个权重
        for u, v in graph.edges():
            if isinstance(graph, IMGraphCpp):
                graph.edges[(u, v)] = random.choice(weight_list)
            else:
                graph.edges[(u, v)]['weight'] = random.choice(weight_list)
    elif edge_weight_type == 'WC':
        # 为每条边设置基于目标节点入度的倒数作为权重
        for u, v in graph.edges():
            if isinstance(graph, IMGraphCpp):
                graph.edges[(u, v)] = 1 / graph.in_degree(v)
            else:
                if graph.is_directed():
                    graph.edges[(u, v)]['weight'] = 1 / graph.in_degree(v)
                else:
                    graph.edges[(u, v)]['weight'] = 1 / graph.degree(v)
    else:
        # 如果edge_weight_type不是上述任何一种，则抛出错误
        raise ValueError('不支持的边权重模型')


def infection_threshold(graph: Union[Graph, DiGraph]):
    """
    计算基于图graph的度分布的感染阈值。

    参数:
        graph (Graph|DiGraph): 网络图对象，可以是有向图或无向图

    返回:
        float: 感染阈值（是1.05倍的阈值）。

    说明:
        该函数首先计算图中所有节点的度之和（k），然后计算所有节点度的平方和（k2）。
        感染阈值计算公式为 k / (k2 - k)，其中k2是度的平方和，k是度的总和。
        这个阈值在某些流行病学模型（如SIR模型）中用于预测疾病传播的临界条件。
    """
    # 计算图中所有节点的度之和
    k = sum(dict(graph.degree()).values())

    # 计算图中所有节点度的平方和
    # 使用map函数和lambda表达式将每个度值平方，然后求和
    k2 = sum(map(lambda x: x ** 2, dict(graph.degree()).values()))

    # 计算并返回感染阈值
    beta = k / (k2 - k)
    return beta + 0.05 * beta


def topk(res_dict: dict, k: int, largest=True):
    """
    从字典中返回具有最大（或最小）k个值的键的列表。

    参数:
        res_dict (dict): 输入的字典，其值用于比较以决定键的排序。

        k (int): 要返回的键的数量。

        largest (bool): 如果为True，则返回具有最大值的k个键；如果为False，则返回具有最小值的k个键。

    返回:
        list: 一个包含k个键的列表，这些键对应于字典中最大（或最小）的值。
    """
    # 使用sorted函数对res_dict的键进行排序
    # key=lambda x: res_dict[x] 指定了排序的依据是字典中对应的值
    # reverse=largest 控制了排序的顺序：largest为True时降序，为False时升序
    sorted_keys = sorted(res_dict.keys(), key=lambda x: res_dict[x], reverse=largest)

    # 通过切片操作[:k]返回前k个排序后的键
    # 这将给出具有最大（或最小）k个值的键的列表
    return sorted_keys[:k]


def truncate_padding(seq, max_len, pad=0):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad] * (max_len - len(seq))
