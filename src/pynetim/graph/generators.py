import random
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import IMGraph


def generate_er_graph(
    n: int,
    p: float,
    directed: bool = True,
    random_seed: Optional[int] = None
) -> "IMGraph":
    """生成 Erdős-Rényi 随机图 (G(n, p) 模型)。

    每对节点之间以概率 p 连接一条边。

    Parameters
    ----------
    n : int
        节点数量。
    p : float
        边连接概率，取值范围 [0, 1]。
    directed : bool, optional
        是否有向图，默认 True。
    random_seed : int, optional
        随机种子，用于可重复性。

    Returns
    -------
    IMGraph
        生成的随机图。

    References
    ----------
    Erdős, P., & Rényi, A. (1959). On random graphs I.
    Publicationes Mathematicae, 6, 290-297.

    Examples
    --------
    >>> from pynetim.graph import generate_er_graph
    >>> g = generate_er_graph(n=100, p=0.1, random_seed=42)
    >>> print(f"节点数: {g.num_nodes}, 边数: {g.num_edges}")
    """
    from .graph import IMGraph

    if random_seed is not None:
        random.seed(random_seed)

    if not 0 <= p <= 1:
        raise ValueError(f"概率 p 必须在 [0, 1] 范围内，当前值: {p}")

    edges = []

    if directed:
        for u in range(n):
            for v in range(n):
                if u != v and random.random() < p:
                    edges.append((u, v))
    else:
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    edges.append((u, v))

    return IMGraph(edges=edges, directed=directed)


def generate_ba_graph(
    n: int,
    m: int,
    directed: bool = True,
    random_seed: Optional[int] = None
) -> "IMGraph":
    """生成 Barabási-Albert 无标度网络。

    通过优先连接机制生成具有幂律度分布的网络。

    Parameters
    ----------
    n : int
        最终节点数量。
    m : int
        每个新节点连接的边数，必须小于 n。
    directed : bool, optional
        是否有向图，默认 True。
    random_seed : int, optional
        随机种子，用于可重复性。

    Returns
    -------
    IMGraph
        生成的无标度网络。

    References
    ----------
    Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks.
    Science, 286(5439), 509-512.

    Examples
    --------
    >>> from pynetim.graph import generate_ba_graph
    >>> g = generate_ba_graph(n=100, m=3, random_seed=42)
    >>> print(f"节点数: {g.num_nodes}, 边数: {g.num_edges}")
    """
    from .graph import IMGraph

    if random_seed is not None:
        random.seed(random_seed)

    if m >= n:
        raise ValueError(f"m ({m}) 必须小于 n ({n})")

    edges = []
    degrees = [0] * n
    total_degree = 0

    for v in range(1, m + 1):
        for u in range(v):
            edges.append((u, v))
            degrees[u] += 1
            degrees[v] += 1
            total_degree += 2

    for v in range(m + 1, n):
        targets = set()
        while len(targets) < m:
            r = random.random() * total_degree
            cumsum = 0
            for u in range(v):
                cumsum += degrees[u]
                if cumsum >= r:
                    if u not in targets:
                        targets.add(u)
                    break

        for u in targets:
            edges.append((u, v))
            degrees[u] += 1
            degrees[v] += 1
            total_degree += 2

    return IMGraph(edges=edges, directed=directed)


def generate_ws_graph(
    n: int,
    k: int,
    beta: float,
    directed: bool = True,
    random_seed: Optional[int] = None
) -> "IMGraph":
    """生成 Watts-Strogatz 小世界网络。

    从环形规则网络开始，以概率 beta 重连边。

    Parameters
    ----------
    n : int
        节点数量。
    k : int
        每个节点连接的邻居数（必须是偶数）。
    beta : float
        重连概率，取值范围 [0, 1]。
        - beta=0: 规则网络
        - beta=1: 随机网络
        - beta~0.1: 小世界网络
    directed : bool, optional
        是否有向图，默认 True。
    random_seed : int, optional
        随机种子，用于可重复性。

    Returns
    -------
    IMGraph
        生成的小世界网络。

    References
    ----------
    Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks.
    Nature, 393(6684), 440-442.

    Examples
    --------
    >>> from pynetim.graph import generate_ws_graph
    >>> g = generate_ws_graph(n=100, k=4, beta=0.1, random_seed=42)
    >>> print(f"节点数: {g.num_nodes}, 边数: {g.num_edges}")
    """
    from .graph import IMGraph

    if random_seed is not None:
        random.seed(random_seed)

    if k % 2 != 0:
        raise ValueError(f"k ({k}) 必须是偶数")

    if not 0 <= beta <= 1:
        raise ValueError(f"beta 必须在 [0, 1] 范围内，当前值: {beta}")

    edges = set()

    half_k = k // 2
    for u in range(n):
        for j in range(1, half_k + 1):
            v = (u + j) % n
            edges.add((u, v))

    edges = list(edges)

    for u in range(n):
        for j in range(1, half_k + 1):
            v = (u + j) % n

            if random.random() < beta:
                edges.remove((u, v))

                new_v = random.choice([x for x in range(n) if x != u and (u, x) not in edges])
                edges.append((u, new_v))

    return IMGraph(edges=edges, directed=directed)
