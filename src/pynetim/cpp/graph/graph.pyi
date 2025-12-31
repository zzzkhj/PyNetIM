from typing import List, Tuple, Dict


class IMGraphCpp:
    """
    图结构类（C++ 后端实现，Python 类型存根）

    本类用于表示有向 / 无向加权图，主要服务于
    影响力传播（IC / LT / SIR 等）与相关算法。

    实际逻辑由 C++ 扩展模块（graph.pyd）实现，
    本文件仅用于类型提示与文档生成。
    """

    # ================== 基本属性 ==================

    #: 图中节点总数（节点编号默认从 0 到 num_nodes - 1）
    num_nodes: int

    #: 图中边的总数
    num_edges: int

    #: 是否为有向图
    directed: bool

    #: 出邻接表（只读暴露）
    #: adj[u] 表示从节点 u 出发可达的所有邻居节点
    adj: List[List[int]]

    #: 入邻接表（只读暴露，仅在有向图下有意义）
    #: rev_adj[u] 表示所有指向节点 u 的前驱节点
    rev_adj: List[List[int]]

    #: 边权重映射
    #: key 为 (u, v)，value 为边 (u → v) 的权重 / 传播概率
    edges: Dict[Tuple[int, int], float]

    # ================== 构造函数 ==================

    def __init__(self, num_nodes: int,
                 edges: List[Tuple[int, int]],
                 weights: List[float] = ...,
                 directed: bool = True) -> None:
        """
        构造一个图对象

        Args
        ----------
        num_nodes : int
            图中节点数量
        edges : List[Tuple[int, int]]
            边列表，每个元素为 (u, v)
        weights : List[float], optional
            对应边的权重列表；若不提供，则所有边权重默认为 1.0
        directed : bool, default=True
            是否构造为有向图
        """
        ...

    # ================== 构图操作 ==================

    def add_edge(self, u: int, v: int, w: float = 1.0) -> None:
        """
        添加一条边 (u → v)

        Args
        ----------
        u : int
            起始节点
        v : int
            终止节点
        w : float, default=1.0
            边权重（在传播模型中表示激活概率）
        """
        ...

    def add_edges(
        self,
        edges: List[Tuple[int, int]],
        weights: List[float] = ...
    ) -> None:
        """
        批量添加多条边

        Args
        ----------
        edges : List[Tuple[int, int]]
            边列表，每个元素为 (u, v)
        weights : List[float], optional
            对应边的权重列表；若不提供，则所有边权重默认为 1.0
        """
        ...

    def update_edge_weight(self, u: int, v: int, w: float) -> None:
        """
        更新已存在边的权重

        Args
        ----------
        u : int
            起始节点
        v : int
            终止节点
        w : float
            新的边权重
        """
        ...
    # ================== 删除边操作 ==================
    def remove_edge(self, u: int, v: int) -> None:
        """
        删除一条边 (u → v)

        Args
        ----------
        u : int
            起始节点
        v : int
            终止节点
        """
        ...

    def remove_edges(
        self,
        edges: List[Tuple[int, int]]
    ) -> None:
        """
        批量删除多条边

        Args
        ----------
        edges : List[Tuple[int, int]]
            边列表，每个元素为 (u, v)
        """
        ...
    # ================== 查询接口 ==================

    def out_neighbors(self, u: int) -> List[int]:
        """
        获取节点 u 的出邻居节点列表

        Args
        ----------
        u : int
            节点编号

        Returns
        -------
        List[int]
            所有从 u 出发可达的邻居节点
        """
        ...

    def in_neighbors(self, u: int) -> List[int]:
        """
        获取节点 u 的入邻居节点列表

        Args
        ----------
        u : int
            节点编号

        Returns
        -------
        List[int]
            所有指向 u 的前驱节点
        """
        ...

    def out_degree(self, u: int) -> int:
        """
        返回节点 u 的出度
        """
        ...

    def in_degree(self, u: int) -> int:
        """
        返回节点 u 的入度
        """
        ...

    def degree(self, u: int) -> int:
        """
        返回节点 u 的度

        对于有向图，通常等于 in_degree + out_degree；
        对于无向图，等于邻接节点数量。
        """
        ...
    def get_adj_list(self) -> List[List[int]]:
        """
        返回图的出邻接表（深拷贝或只读视图，取决于 C++ 实现）

        Returns
        -------
        List[List[int]]
            邻接表表示的图结构
        """
        ...

    def get_adj_matrix(self) -> List[List[int]]:
        """
        返回图的出邻接矩阵（深拷贝或只读视图，取决于 C++ 实现）

        Returns
        -------
        List[List[int]]
            邻接表表示的图结构
        """
        ...
