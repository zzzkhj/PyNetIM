import random
from multiprocessing import Pool, cpu_count
import copy
import statistics
import time

from decorator import Timer
from .base_diffusion_model import BaseDiffusionModel


def __simulate_multi_round(diffusion_model: BaseDiffusionModel, round: int, seed: int = None):
    """
    在单个进程中执行多轮蒙特卡洛模拟。

    Args:
        diffusion_model (BaseDiffusionModel): 扩散模型实例
        round (int): 该进程需要执行的模拟轮数
        seed (int, optional): 随机种子的基础值，默认为None

    Returns:
        float: 该进程所有模拟轮次的平均激活节点数
    """
    count = 0
    for i in range(round):
        random.seed(seed + i if seed is not None else None)
        diffusion_model.reset()
        result = diffusion_model.diffusion()
        count += len(result)
    return count / round


def run_monte_carlo_diffusion(
        diffusion_model: BaseDiffusionModel,
        round: int,
        multi_process: bool = False,
        processes: int = None,
        seed: int = None,
):
    """
    执行蒙特卡洛模拟扩散过程，支持单进程和多进程模式，并可输出所用时间。

    Args:
        diffusion_model (BaseDiffusionModel): 扩散模型实例
        round (int): 总模拟轮数
        multi_process (bool, optional): 是否启用多进程模式，默认为False
        processes (int, optional): 多进程模式下的进程数，为None时使用CPU核心数
        seed (int, optional): 随机种子的基础值，默认为None

    Returns:
        float: 所有模拟轮次的平均激活节点数
    """

    if multi_process:
        if processes is None:
            processes = cpu_count()

        # 每个进程执行 round / processes 次模拟
        rounds_per_worker = int(round / processes)
        with Pool(processes=processes) as pool:
            # 每个进程需要的参数
            args = [
                (diffusion_model, rounds_per_worker, i)
                for i in range(processes)
            ]
            results = pool.starmap(__simulate_multi_round, args)

        avg_activated = statistics.mean(results)
    else:
        # 单进程模式
        avg_activated = __simulate_multi_round(diffusion_model, round, seed)

    return avg_activated
