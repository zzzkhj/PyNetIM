"""C++ 模块随机种子管理。

本模块仅控制 PyNetIM 内部 C++ 模块（扩散模型、RIS 算法等）的随机种子。
C++ 模块通过 pybind11 回调 Python 端获取全局种子。

Python 层面（Python random、NumPy、PyTorch 等）的随机种子由用户自行管理，
本模块不做干预。

Example:
    >>> import pynetim
    >>> pynetim.random.seed(42)
    >>> # C++ 模块的随机操作将可复现
"""

from typing import Optional

_global_seed: Optional[int] = None


def seed(seed_value: int) -> None:
    """设置 C++ 模块的全局随机种子。

    C++ 模块（扩散模型、RIS 算法等）在调用时会通过 Python 回调获取此种子。
    仅影响 PyNetIM 内部 C++ 模块的随机行为，不影响 Python random、NumPy、PyTorch 等。

    Args:
        seed_value: 随机种子值。

    Example:
        >>> import pynetim
        >>> pynetim.random.seed(42)
        >>> # C++ 模块的随机操作将可复现
    """
    global _global_seed
    _global_seed = seed_value


def clear_seed() -> None:
    """清除 C++ 模块的全局随机种子。

    清除后，C++ 模块将使用真随机。
    """
    global _global_seed
    _global_seed = None


def get_random_seed() -> Optional[int]:
    """获取当前全局随机种子。

    C++ 模块通过 pybind11 回调此函数获取全局种子。

    Returns:
        int | None: 当前种子值，未设置时返回 None。
    """
    return _global_seed


def has_seed() -> bool:
    """检查是否已设置全局随机种子。

    Returns:
        bool: 已设置返回 True，否则返回 False。
    """
    return _global_seed is not None


__all__ = ['seed', 'clear_seed', 'get_random_seed', 'has_seed']
