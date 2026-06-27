import re
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np

from pynetim.graph import IMGraph



def compute_sir_beta(
    graph: "IMGraph",
    gamma: float = 0.1,
    c: float = 1.3
) -> Tuple[float, float]:
    """计算 SIR 模型的感染率 β。

    基于配置模型近似 (Configuration Model Approximation) 计算感染率。

    公式:
        β_c = γ · <k> / <k²>
        β = c · β_c

    其中:
        - <k>: 平均度
        - <k²>: 度的平方均值
        - γ: 恢复率
        - c: 调节系数

    Args:
        graph: 图对象。
        gamma: 恢复率，默认 0.1。
        c: 调节系数，默认 1.3。
            - c < 1: 不爆发
            - c = 1: 临界状态
            - c > 1: 爆发

    Returns:
        tuple[float, float]: (beta, beta_c): 感染率和临界感染率。

    References:
        Pastor-Satorras, R., Castellano, C., Van Mieghem, P., & Vespignani, A. (2015).
        Epidemic processes in complex networks. Reviews of Modern Physics, 87(3), 925.

    Example:
        >>> from pynetim.utils import generate_er_graph, compute_sir_beta
        >>> g = generate_er_graph(n=100, p=0.1, random_seed=42)
        >>> beta, beta_c = compute_sir_beta(g, gamma=0.1, c=1.3)
        >>> print(f"感染率: {beta:.4f}, 临界感染率: {beta_c:.4f}")
    """
    degrees = np.array(graph.batch_out_degree(list(range(graph.num_nodes))))

    k_mean = degrees.mean()
    k2_mean = (degrees ** 2).mean()

    if k2_mean == 0:
        raise ValueError("图的度平方均值为 0，无法计算感染率")

    beta_c = gamma * k_mean / k2_mean
    beta = c * beta_c

    return beta, beta_c


def load_edgelist(
    filepath: Union[str, Path],
    directed: bool = True,
    renumber: bool = False,
    comment: Optional[str] = None,
    skip_lines: int = 0
) -> "IMGraph":
    """
    从文件读取边列表构造 IMGraph。
    
    文件格式：每行 u v [weight]，分隔符支持空格/制表符/逗号。
    第三列可选，默认权重为 1.0。
    
    Args:
        filepath: 边列表文件路径。
        directed: 是否有向图，默认 True。
        renumber: 是否重编号节点为连续整数，默认 False。
        comment: 注释行前缀，如 '#' 或 '%'，默认 None。
        skip_lines: 跳过文件开头的行数，默认 0。
    
    Returns:
        IMGraph: 构造的图对象。
    
    Example:
        文件内容 (edges.txt):
        0 1 0.5
        1 2 0.8
        2 3

        >>> g = load_edgelist("edges.txt")
        >>> g.num_nodes
        4
        >>> g.num_edges
        3
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    edges = []
    weights = []
    
    split_pattern = re.compile(r'[\s,\t]+')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in range(skip_lines):
            next(f)
        
        for line_num, line in enumerate(f, start=skip_lines + 1):
            line = line.strip()
            
            if not line:
                continue
            
            if comment and line.startswith(comment):
                continue
            
            parts = split_pattern.split(line)
            
            if len(parts) < 2:
                raise ValueError(f"第 {line_num} 行格式错误: {line!r}")
            
            try:
                u = int(parts[0])
                v = int(parts[1])
                w = float(parts[2]) if len(parts) > 2 else 1.0
            except ValueError as e:
                raise ValueError(f"第 {line_num} 行解析错误: {line!r}") from e
            
            edges.append((u, v))
            weights.append(w)
    
    return IMGraph(edges=edges, weights=weights, directed=directed, renumber=renumber)


def save_edgelist(
    graph: "IMGraph",
    filepath: Union[str, Path],
    delimiter: str = "\t",
    include_weight: bool = True
) -> None:
    """
    将 IMGraph 保存为边列表文件。
    
    Args:
        graph: 图对象。
        filepath: 输出文件路径。
        delimiter: 分隔符，默认制表符。
        include_weight: 是否包含权重列，默认 True。
    
    Example:
        >>> g = IMGraph(edges=[(0, 1), (1, 2)], weights=[0.5, 0.8])
        >>> save_edgelist(g, "output.txt")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for u in range(graph.num_nodes):
            for neighbor, weight in graph.out_neighbors_with_weights(u):
                if include_weight:
                    f.write(f"{u}{delimiter}{neighbor}{delimiter}{weight}\n")
                else:
                    f.write(f"{u}{delimiter}{neighbor}\n")


def to_networkx(graph: "IMGraph"):
    """
    将 IMGraph 转换为 networkx.DiGraph 或 networkx.Graph。
    
    Args:
        graph: 图对象。
    
    Returns:
        networkx.DiGraph 或 networkx.Graph: 取决于原图是否有向。
    
    Example:
        >>> g = IMGraph(edges=[(0, 1), (1, 2)], directed=True)
        >>> nx_g = to_networkx(g)
        >>> nx_g.number_of_nodes()
        3
    """
    import networkx as nx
    
    if graph.directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    G.add_nodes_from(range(graph.num_nodes))
    
    all_neighbors = graph.batch_out_neighbors_with_weights(list(range(graph.num_nodes)))
    
    for u, neighbors in enumerate(all_neighbors):
        for neighbor, weight in neighbors:
            G.add_edge(u, neighbor, weight=weight)
    
    return G


def to_igraph(graph: "IMGraph"):
    """
    将 IMGraph 转换为 igraph.Graph。
    
    Args:
        graph: 图对象。
    
    Returns:
        igraph.Graph: igraph 图对象，边权重存储在 'weight' 属性中。
    
    Example:
        >>> g = IMGraph(edges=[(0, 1), (1, 2)], directed=True)
        >>> ig_g = to_igraph(g)
        >>> ig_g.vcount()
        3
    """
    import igraph as ig
    
    edges = []
    weights = []
    
    all_neighbors = graph.batch_out_neighbors_with_weights(list(range(graph.num_nodes)))
    
    for u, neighbors in enumerate(all_neighbors):
        for neighbor, weight in neighbors:
            edges.append((u, neighbor))
            weights.append(weight)
    
    G = ig.Graph(n=graph.num_nodes, edges=edges, directed=graph.directed)
    G.es['weight'] = weights
    
    return G


def to_scipy_sparse(graph: "IMGraph", format: str = "csr"):
    """
    将 IMGraph 转换为 scipy 稀疏矩阵。
    
    Args:
        graph: 图对象。
        format: 稀疏矩阵格式，可选 'csr', 'csc', 'coo', 'lil', 'dok'，默认 'csr'。
    
    Returns:
        scipy.sparse.spmatrix: 稀疏邻接矩阵，matrix[i, j] 表示边 (i, j) 的权重。
    
    Example:
        >>> g = IMGraph(edges=[(0, 1), (1, 2)], weights=[0.5, 0.8], directed=True)
        >>> mat = to_scipy_sparse(g)
        >>> mat.toarray()
        array([[0. , 0.5, 0. ],
               [0. , 0. , 0.8],
               [0. , 0. , 0. ]])
    """
    import scipy.sparse as sp
    
    sparse_data = graph.get_adj_matrix_sparse()
    
    if not sparse_data:
        return sp.coo_matrix((graph.num_nodes, graph.num_nodes))
    
    rows = [e[0] for e in sparse_data]
    cols = [e[1] for e in sparse_data]
    data = [e[2] for e in sparse_data]
    
    mat = sp.coo_matrix((data, (rows, cols)), shape=(graph.num_nodes, graph.num_nodes))
    
    format = format.lower()
    if format == "csr":
        return mat.tocsr()
    elif format == "csc":
        return mat.tocsc()
    elif format == "lil":
        return mat.tolil()
    elif format == "dok":
        return mat.todok()
    else:
        return mat


def to_pyg(graph: "IMGraph"):
    """
    将 IMGraph 转换为 PyTorch Geometric Data 对象。
    
    Args:
        graph: 图对象。
    
    Returns:
        torch_geometric.data.Data: PyG Data 对象，包含 edge_index 和 edge_attr (权重)。
    
    Example:
        >>> g = IMGraph(edges=[(0, 1), (1, 2)], weights=[0.5, 0.8], directed=True)
        >>> pyg_g = to_pyg(g)
        >>> pyg_g.num_nodes
        3
    """
    import torch
    from torch_geometric.data import Data
    
    sparse_data = graph.get_adj_matrix_sparse()
    
    if not sparse_data:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor([[e[0], e[1]] for e in sparse_data], dtype=torch.long).t()
        edge_attr = torch.tensor([[e[2]] for e in sparse_data], dtype=torch.float)
    
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=graph.num_nodes)


def from_networkx(graph, weight_attr: str = "weight", default_weight: float = 1.0) -> "IMGraph":
    """
    从 networkx.Graph 或 networkx.DiGraph 创建 IMGraph。

    Args:
        graph: networkx 图对象。
        weight_attr: 边权重属性名，默认 'weight'。
        default_weight: 默认权重值，当边没有权重属性时使用，默认 1.0。

    Returns:
        IMGraph: 转换后的图对象。

    Example:
        >>> import networkx as nx
        >>> nx_g = nx.DiGraph()
        >>> nx_g.add_edge(0, 1, weight=0.5)
        >>> nx_g.add_edge(1, 2, weight=0.8)
        >>> g = from_networkx(nx_g)
        >>> g.num_nodes
        3
    """
    edges = []
    weights = []

    node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}

    for u, v, data in graph.edges(data=True):
        edges.append((node_mapping[u], node_mapping[v]))
        weights.append(data.get(weight_attr, default_weight))

    return IMGraph(edges=edges, weights=weights, directed=graph.is_directed(), renumber=False)


def from_igraph(graph, weight_attr: str = "weight", default_weight: float = 1.0) -> "IMGraph":
    """
    从 igraph.Graph 创建 IMGraph。

    Args:
        graph: igraph 图对象。
        weight_attr: 边权重属性名，默认 'weight'。
        default_weight: 默认权重值，当边没有权重属性时使用，默认 1.0。

    Returns:
        IMGraph: 转换后的图对象。

    Example:
        >>> import igraph as ig
        >>> ig_g = ig.Graph(n=3, edges=[(0, 1), (1, 2)], directed=True)
        >>> ig_g.es['weight'] = [0.5, 0.8]
        >>> g = from_igraph(ig_g)
        >>> g.num_nodes
        3
    """
    edges = []
    weights = []

    if 'weight' in graph.edge_attributes():
        weights = graph.es[weight_attr]
    else:
        weights = [default_weight] * graph.ecount()

    edges = [e.tuple for e in graph.es]

    return IMGraph(edges=edges, weights=weights, directed=graph.is_directed(), renumber=False)


def from_scipy_sparse(matrix, directed: bool = True) -> "IMGraph":
    """
    从 scipy 稀疏矩阵创建 IMGraph。

    Args:
        matrix: scipy 稀疏矩阵，matrix[i, j] 表示边 (i, j) 的权重。
        directed: 是否有向图，默认 True。

    Returns:
        IMGraph: 转换后的图对象。

    Example:
        >>> import scipy.sparse as sp
        >>> mat = sp.csr_matrix([[0, 0.5, 0], [0, 0, 0.8], [0, 0, 0]])
        >>> g = from_scipy_sparse(mat, directed=True)
        >>> g.num_nodes
        3
    """
    coo = matrix.tocoo()

    edges = list(zip(coo.row.tolist(), coo.col.tolist()))
    weights = coo.data.tolist()

    return IMGraph(edges=edges, weights=weights, directed=directed, renumber=False)


def from_pyg(data, default_weight: float = 1.0) -> "IMGraph":
    """
    从 PyTorch Geometric Data 对象创建 IMGraph。

    Args:
        data: torch_geometric.data.Data 对象。
        default_weight: 默认权重值，当 Data 没有 edge_attr 时使用，默认 1.0。

    Returns:
        IMGraph: 转换后的图对象。

    Example:
        >>> import torch
        >>> from torch_geometric.data import Data
        >>> edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
        >>> edge_attr = torch.tensor([[0.5], [0.8]], dtype=torch.float)
        >>> pyg_data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
        >>> g = from_pyg(pyg_data)
        >>> g.num_nodes
        3
    """
    edge_index = data.edge_index.cpu().numpy()

    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        weights = data.edge_attr.cpu().numpy().flatten().tolist()
    else:
        weights = [default_weight] * len(edges)

    directed = True

    return IMGraph(edges=edges, weights=weights, directed=directed, renumber=False)
