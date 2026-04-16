from typing import List, Tuple, Dict, Set, Union, TYPE_CHECKING

if TYPE_CHECKING:
    pass

class Edge:
    """边对象。
    
    Attributes:
        to: 目标节点。
        weight: 边权重。
    """
    to: int
    weight: float

class IMGraph:
    """图结构类。
    
    提供高效的 C++ 图实现，支持有向/无向图、边权重等特性。
    
    Attributes:
        num_nodes: 节点数量。
        num_edges: 边数量。
        directed: 是否为有向图。
        edges: 边列表及权重。
        original_to_internal: 原始节点 ID 到内部 ID 的映射。
        internal_to_original: 内部节点 ID 到原始 ID 的映射。
    """
    num_nodes: int
    num_edges: int
    directed: bool
    edges: Dict[Tuple[int, int], float]
    original_to_internal: Dict[int, int]
    internal_to_original: List[int]
    
    def __init__(
        self,
        edges: List[Tuple[int, int]],
        weights: Union[float, List[float]] = 1.0,
        directed: bool = True,
        renumber: bool = True
    ) -> None:
        """从边列表构建图。
        
        传入的边列表必须是从 0 开始连续编号节点，若不是请设置 renumber=True。
        
        Args:
            edges: 边列表，每个元素为 (u, v) 元组。
            weights: 边权重。若为列表则指定每条边的权重；若为浮点数则统一权重。默认为 1.0。
            directed: 是否为有向图，默认为 True。
            renumber: 是否重新编号节点。若为 True，将节点重新编号为从 0 开始的连续整数。默认为 True。
        """
        ...
    
    def add_edge(self, u: int, v: int, w: float = 1.0) -> None:
        """添加带权边。
        
        Args:
            u: 源节点。
            v: 目标节点。
            w: 边权重，默认为 1.0。
        """
        ...
    
    def add_edges(self, edges: List[Tuple[int, int]], weights: List[float] = ...) -> None:
        """批量添加边。
        
        Args:
            edges: 边列表，每个元素为 (u, v) 元组。
            weights: 边权重列表。
        """
        ...
    
    def update_edge_weight(self, u: int, v: int, w: float) -> None:
        """更新已有边的权重。
        
        Args:
            u: 源节点。
            v: 目标节点。
            w: 新的边权重。
        """
        ...
    
    def remove_edge(self, u: int, v: int) -> None:
        """删除边。
        
        Args:
            u: 源节点。
            v: 目标节点。
        """
        ...
    
    def remove_edges(self, edges: List[Tuple[int, int]]) -> None:
        """批量删除边。
        
        Args:
            edges: 边列表，每个元素为 (u, v) 元组。
        """
        ...
    
    def out_neighbors(self, u: int) -> List[int]:
        """返回节点的出边邻居。
        
        Args:
            u: 节点 ID。
        
        Returns:
            List[int]: 出边邻居节点 ID 列表。
        """
        ...
    
    def out_neighbors_with_weights(self, u: int) -> List[Tuple[int, float]]:
        """返回节点的出边邻居及权重。
        
        Args:
            u: 节点 ID。
        
        Returns:
            List[Tuple[int, float]]: (邻居, 权重) 元组列表。
        """
        ...
    
    def out_neighbors_arrays(self, u: int) -> Tuple[List[int], List[float]]:
        """返回节点的出边邻居（数组格式）。
        
        Args:
            u: 节点 ID。
        
        Returns:
            Tuple[List[int], List[float]]: 两个并行数组 (targets, weights)，O(1) 访问每个元素。
        """
        ...
    
    def in_neighbors(self, u: int) -> List[int]:
        """返回节点的入边邻居。
        
        Args:
            u: 节点 ID。
        
        Returns:
            List[int]: 入边邻居节点 ID 列表。
        """
        ...
    
    def out_degree(self, u: int) -> int:
        """返回节点的出度。
        
        Args:
            u: 节点 ID。
        
        Returns:
            int: 出度。
        """
        ...
    
    def in_degree(self, u: int) -> int:
        """返回节点的入度。
        
        Args:
            u: 节点 ID。
        
        Returns:
            int: 入度。
        """
        ...
    
    def degree(self, u: int) -> int:
        """返回节点的度数。
        
        Args:
            u: 节点 ID。
        
        Returns:
            int: 度数。
        """
        ...
    
    def get_all_degrees(self) -> List[int]:
        """返回所有节点的度数列表。
        
        Returns:
            List[int]: 度数列表。
        """
        ...
    
    def get_all_in_degrees(self) -> List[int]:
        """返回所有节点的入度列表。
        
        Returns:
            List[int]: 入度列表。
        """
        ...
    
    def get_all_out_degrees(self) -> List[int]:
        """返回所有节点的出度列表。
        
        Returns:
            List[int]: 出度列表。
        """
        ...
    
    def batch_out_degree(self, nodes: List[int]) -> List[int]:
        """批量返回指定节点的出度。
        
        Args:
            nodes: 节点 ID 列表。
        
        Returns:
            List[int]: 出度列表。
        """
        ...
    
    def batch_in_degree(self, nodes: List[int]) -> List[int]:
        """批量返回指定节点的入度。
        
        Args:
            nodes: 节点 ID 列表。
        
        Returns:
            List[int]: 入度列表。
        """
        ...
    
    def batch_degree(self, nodes: List[int]) -> List[int]:
        """批量返回指定节点的度数。
        
        Args:
            nodes: 节点 ID 列表。
        
        Returns:
            List[int]: 度数列表。
        """
        ...
    
    def batch_out_neighbors(self, nodes: List[int]) -> List[List[int]]:
        """批量返回指定节点的出边邻居。
        
        Args:
            nodes: 节点 ID 列表。
        
        Returns:
            List[List[int]]: 每个节点的出边邻居节点 ID 列表。
        """
        ...
    
    def batch_out_neighbors_with_weights(self, nodes: List[int]) -> List[List[Tuple[int, float]]]:
        """批量返回指定节点的出边邻居及权重。
        
        Args:
            nodes: 节点 ID 列表。
        
        Returns:
            List[List[Tuple[int, float]]]: 每个节点的 (邻居, 权重) 元组列表。
        """
        ...
    
    def batch_get_edge_weight(self, edges: List[Tuple[int, int]], default_value: float = 0.0, raise_on_missing: bool = False) -> List[float]:
        """批量返回指定边的权重。
        
        Args:
            edges: 边列表，每个元素为 (u, v) 元组。
            default_value: 边不存在时的默认返回值，默认为 0.0。当 raise_on_missing=True 时忽略。
            raise_on_missing: 边不存在时是否抛出异常，默认为 False。
        
        Returns:
            List[float]: 权重列表。
        
        Raises:
            RuntimeError: 当 raise_on_missing=True 且边不存在时。
        """
        ...
    
    def get_adj_list(self) -> List[List[Edge]]:
        """返回完整邻接表。
        
        Returns:
            List[List[Edge]]: Edge 对象列表的列表。
        """
        ...
    
    def get_adj_list_py(self) -> List[List[Tuple[int, float]]]:
        """返回 Python 友好格式的邻接表。
        
        Returns:
            List[List[Tuple[int, float]]]: (邻居, 权重) 元组列表的列表。
        """
        ...
    
    def get_adj_matrix(self) -> List[List[float]]:
        """返回稠密邻接矩阵。
        
        Returns:
            List[List[float]]: 邻接矩阵。
        """
        ...
    
    def get_adj_matrix_sparse(self) -> List[Tuple[int, int, float]]:
        """返回稀疏邻接矩阵。
        
        Returns:
            List[Tuple[int, int, float]]: (u, v, 权重) 元组列表。
        """
        ...
    
    def get_edge_weight(self, u: int, v: int) -> float:
        """获取边的权重。若边不存在则抛出异常。
        
        Args:
            u: 源节点。
            v: 目标节点。
        
        Returns:
            float: 边权重。
        
        Raises:
            RuntimeError: 边不存在。
        """
        ...
    
    def has_edge(self, u: int, v: int) -> bool:
        """检查边是否存在。
        
        Args:
            u: 源节点。
            v: 目标节点。
        
        Returns:
            bool: 边存在返回 True，否则返回 False。
        """
        ...
    
    def __repr__(self) -> str: ...
