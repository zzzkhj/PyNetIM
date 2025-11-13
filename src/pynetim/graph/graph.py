from __future__ import annotations

from networkx import Graph, DiGraph

from ..utils import set_edge_weight


class IMGraph:
    """
    影响最大化图类，用于封装网络图结构和边权重设置。

    该类提供了对NetworkX图对象的封装，支持设置不同类型的边权重，
    并提供了便捷的图属性访问接口。

    Attributes:
        graph (Graph | DiGraph): networkx图对象，可以是有向图或无向图
        direction (bool): 指示图是否有向
        number_of_nodes (int): 图中节点总数
        number_of_edges (int): 图中边总数
        edge_weight_type (str): 边权重类型
    """

    def __init__(self, graph: Graph | DiGraph, edge_weight_type: str, constant_weight: float = None):
        """
        初始化IMGraph实例。

        Args:
            graph (Graph | DiGraph): networkx图对象，可以是有向图或无向图
            edge_weight_type (str): 边权重类型，支持' CONSTANT '、' TV '、' WC '等模式
            constant_weight (float, optional): 当使用常量权重模式时的权重值
        """
        self.nx_graph = graph
        self.direction = graph.is_directed()
        self.number_of_nodes = graph.number_of_nodes()
        self.number_of_edges = graph.number_of_edges()
        self.edge_weight_type = edge_weight_type

        set_edge_weight(self.nx_graph, edge_weight_type, constant_weight)

    @property
    def nodes(self):
        """
        获取图中所有节点。

        Returns:
            NodeView: 图中所有节点的视图
        """
        return self.nx_graph.nodes

    @property
    def edges(self):
        """
        获取图中所有边。

        Returns:
            EdgeView: 图中所有边的视图
        """
        return self.nx_graph.edges

    def neighbors(self, node):
        """
        获取指定节点的邻居节点。

        Args:
            node: 节点标识符

        Returns:
            iterator: 指定节点的所有邻居节点迭代器
        """
        return self.nx_graph.neighbors(node)

    def in_neighbors(self, node):
        """
        获取指定节点的入邻居节点。

        对于有向图，返回前驱节点；对于无向图，返回所有邻居节点。

        Args:
            node: 节点标识符

        Returns:
            iterator: 指定节点的入邻居节点迭代器
        """
        return self.nx_graph.predecessors(node) if self.direction else self.nx_graph.neighbors(node)

    def out_neighbors(self, node):
        """
        获取指定节点的出邻居节点。（同neighbors方法）

        Args:
            node: 节点标识符

        Returns:
            iterator: 指定节点的出邻居节点迭代器
        """
        return self.nx_graph.neighbors(node)

    @property
    def in_degree(self):
        """
        获取图的入度。

        Returns:
            int: 图的入度视图
        """
        return self.nx_graph.in_degree if self.direction else self.nx_graph.degree()

    @property
    def out_degree(self):
        """
        获取图的出度。

        Returns:
            int: 图的出度视图
        """
        return self.nx_graph.out_degree if self.direction else self.nx_graph.degree()

    def degree(self):
        """
        获取图的度。

        Returns:
            int: 图的度视图
        """
        return self.nx_graph.degree()

    def __str__(self):
        """
        返回图对象的字符串表示。

        Returns:
            str: 图对象的字符串描述
        """
        return f'{self.nx_graph} and edge_weight_type: {self.edge_weight_type}'

    def __repr__(self):
        """
        返回图对象的详细字符串表示。

        Returns:
            str: 图对象的详细字符串描述
        """
        return self.__str__()
