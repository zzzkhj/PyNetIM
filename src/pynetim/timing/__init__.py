# -*- coding: utf-8 -*-
"""时间测量工具模块。

提供算法运行时间测量的多种方式。
"""

from typing import List, Dict, Callable, Any, Optional, Set, TYPE_CHECKING
from functools import wraps
import time
import numpy as np

if TYPE_CHECKING:
    from ..graph import IMGraph


def measure_time(func: Callable) -> Callable:
    """装饰器：测量函数或方法的运行时间。
    
    可以用于类方法或普通函数，支持C++绑定方法。
    
    Args:
        func: 要测量的函数或方法。
    
    Returns:
        Callable: 包装后的函数。
    
    Example:
        >>> from pynetim.timing import measure_time
        >>> 
        >>> @measure_time
        >>> def my_algorithm(graph, k):
        ...     # 算法实现
        ...     return seeds
        >>> 
        >>> seeds, runtime = my_algorithm(graph, k=10)
        >>> print(f"Runtime: {runtime:.4f}s")
        
        或者用于类方法：
        >>> class MyAlgorithm:
        ...     @measure_time
        ...     def run(self, k):
        ...         return self._find_seeds(k)
    
    Note:
        装饰器会修改函数的返回值，将原始返回值和运行时间打包成元组返回。
        对于类方法，装饰器会自动处理 self 参数。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        runtime = end_time - start_time
        
        return result, runtime
    
    return wrapper


def measure_runtime(
    func: Callable,
    *args,
    **kwargs
) -> tuple:
    """测量函数运行时间。
    
    Args:
        func: 要测量的函数。
        *args: 函数位置参数。
        **kwargs: 函数关键字参数。
    
    Returns:
        tuple: (函数返回值, 运行时间秒数)
    
    Example:
        >>> from pynetim.timing import measure_runtime
        >>> result, runtime = measure_runtime(algorithm.run, k=10)
        >>> print(f"Runtime: {runtime:.4f}s")
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    return result, end_time - start_time


def measure_runtime_multiple_runs(
    func: Callable,
    num_runs: int = 10,
    *args,
    **kwargs
) -> Dict[str, float]:
    """多次运行函数并统计运行时间。
    
    Args:
        func: 要测量的函数。
        num_runs: 运行次数，默认 10。
        *args: 函数位置参数。
        **kwargs: 函数关键字参数。
    
    Returns:
        Dict[str, float]: 包含以下统计量：
            - mean: 平均运行时间
            - std: 标准差
            - min: 最小运行时间
            - max: 最大运行时间
            - median: 中位数
            - total: 总运行时间
    
    Example:
        >>> from pynetim.timing import measure_runtime_multiple_runs
        >>> stats = measure_runtime_multiple_runs(algorithm.run, num_runs=5, k=10)
        >>> print(f"Mean runtime: {stats['mean']:.4f}s")
    """
    runtimes = []
    
    for _ in range(num_runs):
        _, runtime = measure_runtime(func, *args, **kwargs)
        runtimes.append(runtime)
    
    return {
        'mean': float(np.mean(runtimes)),
        'std': float(np.std(runtimes)),
        'min': float(np.min(runtimes)),
        'max': float(np.max(runtimes)),
        'median': float(np.median(runtimes)),
        'total': float(np.sum(runtimes))
    }


class AlgorithmTimer:
    """算法运行时间测量器。
    
    用于测量算法类实例的运行时间，支持多次运行统计。
    
    支持的算法类型：
        - 启发式算法（DegreeDiscount, SingleDiscount等）：只需 k 参数
        - 贪婪算法（Greedy, CELF）：需要 k, mc_rounds 等参数
        - RIS算法（IMM, TIM, OPIM）：需要 k 参数，部分需要 num_rr_sets
    
    Example:
        >>> from pynetim.timing import AlgorithmTimer
        >>> from pynetim import DegreeDiscountAlgorithm
        >>> 
        >>> # 创建算法实例
        >>> algorithm = DegreeDiscountAlgorithm(graph)
        >>> 
        >>> # 创建计时器
        >>> timer = AlgorithmTimer(algorithm)
        >>> 
        >>> # 单次运行（启发式算法）
        >>> seeds, runtime = timer.run(k=10)
        >>> 
        >>> # 单次运行（贪婪算法）
        >>> seeds, runtime = timer.run(k=10, mc_rounds=1000, random_seed=42)
        >>> 
        >>> # 多次运行统计
        >>> stats = timer.run_multiple(k=10, num_runs=5)
        >>> print(f"Mean runtime: {stats['mean']:.4f}s")
    """
    
    def __init__(self, algorithm_instance):
        """初始化计时器。
        
        Args:
            algorithm_instance: 算法类实例，需要有 run() 方法。
        """
        self.algorithm = algorithm_instance
        self._runtimes = []
    
    def run(self, **kwargs) -> tuple:
        """运行算法并测量时间。
        
        Args:
            **kwargs: 传递给算法 run() 方法的参数。
                     必须包含 k 参数（种子节点数量）。
                     
                     常用参数：
                     - k: 种子节点数量（必填）
                     - mc_rounds: 蒙特卡洛模拟次数（贪婪算法）
                     - random_seed: 随机数种子
                     - use_multithread: 是否启用多线程
                     - num_rr_sets: RR集合数量（RIS算法）
        
        Returns:
            tuple: (种子节点集合, 运行时间)
        
        Example:
            >>> # 启发式算法
            >>> seeds, runtime = timer.run(k=10)
            
            >>> # 贪婪算法
            >>> seeds, runtime = timer.run(k=10, mc_rounds=1000, random_seed=42)
            
            >>> # OPIM 算法
            >>> seeds, runtime = timer.run(k=10, num_rr_sets=1000)
        """
        start_time = time.perf_counter()
        result = self.algorithm.run(**kwargs)
        end_time = time.perf_counter()
        
        runtime = end_time - start_time
        self._runtimes.append(runtime)
        
        return result, runtime
    
    def run_multiple(self, num_runs: int = 10, **kwargs) -> Dict[str, float]:
        """多次运行算法并统计运行时间。
        
        Args:
            num_runs: 运行次数，默认 10。
            **kwargs: 传递给算法 run() 方法的参数。
                     必须包含 k 参数（种子节点数量）。
        
        Returns:
            Dict[str, float]: 包含以下统计量：
                - mean: 平均运行时间
                - std: 标准差
                - min: 最小运行时间
                - max: 最大运行时间
                - median: 中位数
                - total: 总运行时间
        
        Example:
            >>> stats = timer.run_multiple(k=10, num_runs=5)
            >>> print(f"Mean: {stats['mean']:.4f}s, Std: {stats['std']:.4f}s")
        """
        runtimes = []
        
        for _ in range(num_runs):
            _, runtime = self.run(**kwargs)
            runtimes.append(runtime)
        
        return {
            'mean': float(np.mean(runtimes)),
            'std': float(np.std(runtimes)),
            'min': float(np.min(runtimes)),
            'max': float(np.max(runtimes)),
            'median': float(np.median(runtimes)),
            'total': float(np.sum(runtimes))
        }
    
    @property
    def last_runtime(self) -> Optional[float]:
        """获取最后一次运行的运行时间。
        
        Returns:
            Optional[float]: 运行时间（秒），如果没有运行过则返回 None。
        """
        return self._runtimes[-1] if self._runtimes else None
    
    @property
    def all_runtimes(self) -> List[float]:
        """获取所有运行时间记录。
        
        Returns:
            List[float]: 运行时间列表。
        """
        return self._runtimes.copy()
    
    def clear(self):
        """清空运行时间记录。"""
        self._runtimes.clear()


def compare_algorithms_runtime(
    algorithms: Dict[str, Any],
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """比较多个算法的运行时间。
    
    Args:
        algorithms: 算法名称到算法实例的映射。
        **kwargs: 传递给算法 run() 方法的参数。
                 必须包含 k 参数（种子节点数量）。
    
    Returns:
        Dict[str, Dict[str, Any]]: 每个算法的运行结果和时间。
            格式：{算法名: {'seeds': 种子集合, 'runtime': 运行时间}}
    
    Example:
        >>> from pynetim.timing import compare_algorithms_runtime
        >>> from pynetim import DegreeDiscountAlgorithm, CELFAlgorithm
        >>> 
        >>> algorithms = {
        ...     'degree_discount': DegreeDiscountAlgorithm(graph),
        ...     'celf': CELFAlgorithm(graph)
        ... }
        >>> results = compare_algorithms_runtime(algorithms, k=10)
        >>> for name, data in results.items():
        ...     print(f"{name}: {data['runtime']:.4f}s, {len(data['seeds'])} seeds")
    """
    results = {}
    
    for name, algorithm in algorithms.items():
        timer = AlgorithmTimer(algorithm)
        seeds, runtime = timer.run(**kwargs)
        
        results[name] = {
            'seeds': seeds,
            'runtime': runtime
        }
    
    return results
