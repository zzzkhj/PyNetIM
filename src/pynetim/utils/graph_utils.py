import re
from pathlib import Path
from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pynetim.graph import IMGraph


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
    
    Parameters
    ----------
    filepath : str or Path
        边列表文件路径
    directed : bool, optional
        是否有向图，默认 True
    renumber : bool, optional
        是否重编号节点为连续整数，默认 False
    comment : str, optional
        注释行前缀，如 '#' 或 '%'，默认 None
    skip_lines : int, optional
        跳过文件开头的行数，默认 0
    
    Returns
    -------
    IMGraph
        构造的图对象
    
    Examples
    --------
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
    from pynetim.graph import IMGraph
    
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
    
    Parameters
    ----------
    graph : IMGraph
        图对象
    filepath : str or Path
        输出文件路径
    delimiter : str, optional
        分隔符，默认制表符
    include_weight : bool, optional
        是否包含权重列，默认 True
    
    Examples
    --------
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
    
    Parameters
    ----------
    graph : IMGraph
        图对象
    
    Returns
    -------
    networkx.DiGraph 或 networkx.Graph
        取决于原图是否有向
    
    Examples
    --------
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
    
    Parameters
    ----------
    graph : IMGraph
        图对象
    
    Returns
    -------
    igraph.Graph
        igraph 图对象，边权重存储在 'weight' 属性中
    
    Examples
    --------
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
    
    Parameters
    ----------
    graph : IMGraph
        图对象
    format : str, optional
        稀疏矩阵格式，可选 'csr', 'csc', 'coo', 'lil', 'dok'，默认 'csr'
    
    Returns
    -------
    scipy.sparse.spmatrix
        稀疏邻接矩阵，matrix[i, j] 表示边 (i, j) 的权重
    
    Examples
    --------
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
    
    Parameters
    ----------
    graph : IMGraph
        图对象
    
    Returns
    -------
    torch_geometric.data.Data
        PyG Data 对象，包含 edge_index 和 edge_attr (权重)
    
    Examples
    --------
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
