from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import IMGraph


def sample_rr_set_ic(graph: 'IMGraph', seed: int | None = None) -> List[int]:
    """采样一个 IC 模型的 RR 集合。

    Args:
        graph: IMGraph 图对象。
        seed: 随机种子，可选。

    Returns:
        List[int]: RR 集合中的节点列表。
    """
    ...


def sample_rr_set_lt(graph: 'IMGraph', seed: int | None = None) -> List[int]:
    """采样一个 LT 模型的 RR 集合。

    Args:
        graph: IMGraph 图对象。
        seed: 随机种子，可选。

    Returns:
        List[int]: RR 集合中的节点列表。
    """
    ...


def generate_rr_sets(
    graph: 'IMGraph',
    num_sets: int,
    model: str = "IC",
    seed: int | None = None
) -> List[List[int]]:
    """生成多个 RR 集合。

    Args:
        graph: IMGraph 图对象。
        num_sets: 要生成的 RR 集合数量。
        model: 传播模型，支持 'IC' 或 'LT'，默认为 'IC'。
        seed: 随机种子，可选。

    Returns:
        List[List[int]]: RR 集合列表。
    """
    ...
