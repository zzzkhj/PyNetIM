import copy
import heapq

from tqdm import tqdm

from graph import IMGraph
from diffusion_model import run_monte_carlo_diffusion, BaseDiffusionModel
from .base_algorithm import BaseAlgorithm


class GreedyAlgorithm(BaseAlgorithm):
    """
    贪婪算法 (Greedy) 选择影响力最大化种子节点

    该算法通过迭代地选择具有最大边际影响力的节点作为种子节点，
    直到选够k个种子节点。每次选择都需要计算所有候选节点的边际增益。

    Attributes:
        graph (IMGraph): 输入图对象
        diffusion_model_class (BaseDiffusionModel): 影响传播模型类
    """

    def __init__(self, graph: IMGraph, diffusion_model: BaseDiffusionModel):
        """
        初始化贪婪算法实例。

        Args:
            graph (IMGraph): 输入图对象
            diffusion_model (BaseDiffusionModel): 影响传播模型类
        """
        super().__init__(graph, diffusion_model)
        self.diffusion_model_class = diffusion_model

    def run(self, k: int, round: int, multi_process=False, processes=None, show_progress=True, seed: int = None):
        """
        运行贪婪算法选择影响力最大化种子节点。

        Args:
            k (int): 种子节点数量
            round (int): 每次计算边际增益的蒙特卡洛模拟次数
            multi_process (bool, optional): 是否启用多进程模式，默认为False
            processes (int, optional): 多进程模式下的进程数，为None时使用默认值
            show_progress (bool, optional): 是否显示进度条，默认为True
            seed (int, optional): 模拟的随机种子，默认为None

        Returns:
            list: 选择的种子节点列表
        """
        seeds = set()
        nodes = set(self.graph.nodes())
        diffusion_model = self.diffusion_model_class(self.graph, list(seeds))

        # 外层循环进度条 - 选择种子节点的进度
        outer_pbar = tqdm(range(k), desc="选择种子节点", disable=not show_progress)

        for i in outer_pbar:
            best_node = None
            best_gain = -1

            node_list = list(nodes - seeds)
            # 内层循环进度条 - 遍历候选节点的进度
            inner_pbar = tqdm(node_list, desc=f"评估候选节点({i+1}/{k})",
                             leave=False, disable=not show_progress)

            for node in inner_pbar:
                # 更新模型种子集合，重置模型
                diffusion_model.reset(list(seeds | {node}))
                avg_influence = run_monte_carlo_diffusion(diffusion_model, round, multi_process, processes, seed)
                # 更新模型种子集合，重置模型
                diffusion_model.reset(list(seeds))
                avg_without = run_monte_carlo_diffusion(diffusion_model, round, multi_process, processes, seed)
                marginal_gain = avg_influence - avg_without

                if marginal_gain > best_gain:
                    best_gain = marginal_gain
                    best_node = node

                # 在进度条上显示当前最佳边际增益
                if show_progress:
                    inner_pbar.set_postfix({'当前最佳增益': f'{best_gain:.4f}'})

            seeds.add(best_node)
            # 更新外层进度条描述信息
            if show_progress:
                outer_pbar.set_description(f"已选中节点: {best_node}")

        self.seeds = list(seeds)
        return list(seeds)


class CELFAlgorithm(BaseAlgorithm):
    """
    CELF算法选择影响力最大化种子节点

    CELF (Cost-Effective Lazy Forward) 算法是贪婪算法的优化版本，
    利用边际增益的子模特性减少计算量，通过优先队列维护节点的边际增益，
    避免重复计算已失效的边际增益。

    Attributes:
        graph (IMGraph): 输入图对象
        diffusion_model_class (BaseDiffusionModel): 影响传播模型类
    """

    def __init__(self, graph: IMGraph, diffusion_model: BaseDiffusionModel):
        """
        初始化CELF算法实例。

        Args:
            graph (IMGraph): 输入图对象
            diffusion_model (BaseDiffusionModel): 影响传播模型类
        """
        super().__init__(graph, diffusion_model)
        self.diffusion_model_class = diffusion_model

    def run(self, k: int, round: int, multi_process=False, processes=None, show_progress=True, seed: int = None):
        """
        运行CELF算法选择影响力最大化种子节点。

        Args:
            k (int): 种子节点数量
            round (int): 蒙特卡洛模拟次数
            multi_process (bool, optional): 是否启用多进程模式，默认为False
            processes (int, optional): 多进程模式下的进程数，为None时使用默认值
            show_progress (bool, optional): 是否显示进度条，默认为True
            seed (int, optional): 模拟的随机数种子，默认为None

        Returns:
            list: 选择的种子节点列表
        """
        seeds = set()
        nodes = set(self.graph.nodes())

        # Initialize marginal gains
        heap = []

        # 计算初始边际增益 - 遍历所有节点
        node_list = list(nodes)
        init_pbar = tqdm(node_list, desc="初始化边际增益", disable=not show_progress)

        diffusion_model = self.diffusion_model_class(self.graph, [])
        for node in init_pbar:
            diffusion_model.reset([node])
            mg = run_monte_carlo_diffusion(diffusion_model, round, multi_process, processes, seed)
            heap.append((-mg, node, 0))  # (负边际增益, 节点, 上次计算时种子集合大小)
        heapq.heapify(heap)

        selected = 0
        # 主循环进度条
        main_pbar = tqdm(range(k), desc="选择种子节点", disable=not show_progress)

        # 创建一个扩散模型实例用于后续计算
        diffusion_model = self.diffusion_model_class(self.graph, list(seeds))

        for _ in main_pbar:
            while True:
                neg_gain, node, last_s_size = heapq.heappop(heap)

                if last_s_size == len(seeds):
                    # 新增种子节点
                    seeds.add(node)
                    selected += 1
                    if show_progress:
                        main_pbar.set_description(f"已选中节点: {node}")
                    break
                else:
                    # 重新计算该节点边际增益
                    diffusion_model.reset(list(seeds | {node}))
                    avg_with = run_monte_carlo_diffusion(diffusion_model, round, multi_process, processes, seed)
                    diffusion_model.reset(list(seeds))
                    avg_without = run_monte_carlo_diffusion(diffusion_model, round, multi_process, processes, seed)
                    marginal_gain = avg_with - avg_without
                    heapq.heappush(heap, (-marginal_gain, node, len(seeds)))  # 修复变量名

        self.seeds = list(seeds)
        return list(seeds)
