from typing import List, Tuple, Dict, Union, overload

@overload
def renumber_edges(edges: List[Tuple[int, int]], return_mapping: bool = False) -> List[Tuple[int, int]]:
    """将边列表重新编号为从 0 开始的连续节点 ID。
    
    Args:
        edges: 边列表，每个元素为 (u, v) 元组，节点 ID 可以是任意整数。
        return_mapping: 是否返回映射关系，默认为 False。
    
    Returns:
        List[Tuple[int, int]]: 重新编号后的边列表。
    """
    ...

@overload
def renumber_edges(edges: List[Tuple[int, int]], return_mapping: bool = True) -> Tuple[List[Tuple[int, int]], Dict[int, int], List[int]]:
    """将边列表重新编号为从 0 开始的连续节点 ID。
    
    Args:
        edges: 边列表，每个元素为 (u, v) 元组，节点 ID 可以是任意整数。
        return_mapping: 是否返回映射关系，默认为 False。
    
    Returns:
        Tuple[List[Tuple[int, int]], Dict[int, int], List[int]]: 
            (重新编号后的边列表, 原始到内部映射, 内部到原始映射)。
    """
    ...
