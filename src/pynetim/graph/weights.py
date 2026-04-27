import random
from typing import Dict, List, Literal, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import IMGraph


def set_edge_weights(
    graph: "IMGraph",
    weight_type: Literal["const", "tv", "uniform", "wc"],
    *,
    const_value: float = 0.01,
    tv_values: List[float] = None,
    uniform_low: float = 0.001,
    uniform_high: float = 0.1,
) -> None:
    """设置图的边权重。

    Args:
        graph: 图对象。
        weight_type: 权重类型：
            - "const": 常数权重
            - "tv": 从列表中随机选择
            - "uniform": 均匀分布随机
            - "wc": WC 模型，权重 = 1 / 入度
        const_value: 常数权重值，仅当 weight_type="const" 时使用，默认 0.01。
        tv_values: 可选权重值列表，仅当 weight_type="tv" 时使用，默认 [0.001, 0.01, 0.1]。
        uniform_low: 均匀分布下界，仅当 weight_type="uniform" 时使用，默认 0.001。
        uniform_high: 均匀分布上界，仅当 weight_type="uniform" 时使用，默认 0.1。

    Raises:
        ValueError: 参数无效时抛出。

    Examples:
        >>> from pynetim.graph import generate_er_graph, set_edge_weights
        >>> g = generate_er_graph(n=100, p=0.1)
        >>> set_edge_weights(g, "const", const_value=0.1)
        >>> set_edge_weights(g, "wc")
    """
    if tv_values is None:
        tv_values = [0.001, 0.01, 0.1]

    weight_type = weight_type.lower()

    if weight_type == "const":
        set_const_weights(graph, const_value)
    elif weight_type == "tv":
        set_tv_weights(graph, tv_values)
    elif weight_type == "uniform":
        set_uniform_weights(graph, uniform_low, uniform_high)
    elif weight_type == "wc":
        set_wc_weights(graph)
    else:
        raise ValueError(
            f"不支持的权重类型: {weight_type}，可选: const, tv, uniform, wc"
        )


def set_const_weights(graph: "IMGraph", value: float) -> None:
    """设置常数边权重。

    Args:
        graph: 图对象。
        value: 常数权重值。

    Raises:
        ValueError: 权重值不在 (0, 1] 范围内时抛出。

    Examples:
        >>> from pynetim.graph import generate_er_graph, set_const_weights
        >>> g = generate_er_graph(n=100, p=0.1)
        >>> set_const_weights(g, value=0.1)
    """
    if not 0 < value <= 1:
        raise ValueError(f"权重值必须在 (0, 1] 范围内，当前值: {value}")

    for (u, v) in graph.edges.keys():
        graph.update_edge_weight(u, v, value)


def set_tv_weights(graph: "IMGraph", values: List[float]) -> None:
    """从列表中随机选择边权重。

    Args:
        graph: 图对象。
        values: 权重值列表。

    Raises:
        ValueError: 权重值列表为空或权重值不在 (0, 1] 范围内时抛出。

    Examples:
        >>> from pynetim.graph import generate_er_graph, set_tv_weights
        >>> g = generate_er_graph(n=100, p=0.1)
        >>> set_tv_weights(g, values=[0.001, 0.01, 0.1])
    """
    if not values:
        raise ValueError("权重值列表不能为空")

    for v in values:
        if not 0 < v <= 1:
            raise ValueError(f"权重值必须在 (0, 1] 范围内，当前值: {v}")

    for (u, v) in graph.edges.keys():
        weight = random.choice(values)
        graph.update_edge_weight(u, v, weight)


def set_uniform_weights(graph: "IMGraph", low: float, high: float) -> None:
    """设置均匀分布边权重。

    Args:
        graph: 图对象。
        low: 均匀分布下界。
        high: 均匀分布上界。

    Raises:
        ValueError: 参数无效时抛出。

    Examples:
        >>> from pynetim.graph import generate_er_graph, set_uniform_weights
        >>> g = generate_er_graph(n=100, p=0.1)
        >>> set_uniform_weights(g, low=0.01, high=0.5)
    """
    if not 0 < low <= 1:
        raise ValueError(f"下界必须在 (0, 1] 范围内，当前值: {low}")
    if not 0 < high <= 1:
        raise ValueError(f"上界必须在 (0, 1] 范围内，当前值: {high}")
    if low >= high:
        raise ValueError(f"下界 ({low}) 必须小于上界 ({high})")

    for (u, v) in graph.edges.keys():
        weight = random.uniform(low, high)
        graph.update_edge_weight(u, v, weight)


def set_wc_weights(graph: "IMGraph") -> None:
    """设置 WC 模型边权重。

    权重 = 1 / 入度。

    Args:
        graph: 图对象。

    Examples:
        >>> from pynetim.graph import generate_er_graph, set_wc_weights
        >>> g = generate_er_graph(n=100, p=0.1)
        >>> set_wc_weights(g)
    """
    in_degrees = graph.get_all_in_degrees()

    for (u, v) in graph.edges.keys():
        in_deg = in_degrees[v]
        weight = 1.0 / in_deg
        graph.update_edge_weight(u, v, weight)


def set_edge_weights_dict(
    graph: "IMGraph", weight_dict: Dict[Tuple[int, int], float]
) -> None:
    """通过字典设置边权重。

    Args:
        graph: 图对象。
        weight_dict: 边权重字典，键为 (u, v) 元组，值为权重。

    Raises:
        ValueError: 权重值无效或边不存在时抛出。

    Examples:
        >>> from pynetim.graph import generate_er_graph, set_edge_weights_dict
        >>> g = generate_er_graph(n=10, p=0.5, random_seed=42)
        >>> weights = {(0, 1): 0.1, (1, 2): 0.2, (2, 3): 0.3}
        >>> set_edge_weights_dict(g, weights)
    """
    for (u, v), weight in weight_dict.items():
        if not 0 < weight <= 1:
            raise ValueError(f"权重值必须在 (0, 1] 范围内，边 ({u}, {v}) 权重: {weight}")
        if not graph.has_edge(u, v):
            raise ValueError(f"边 ({u}, {v}) 不存在于图中")
        graph.update_edge_weight(u, v, weight)
