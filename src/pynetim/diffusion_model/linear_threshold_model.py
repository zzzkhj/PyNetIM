import random

from .run_monte_carlo_diffusion import run_monte_carlo_diffusion
from ..graph import IMGraph
from . import BaseDiffusionModel


class LinearThresholdModel(BaseDiffusionModel):
    """
    线性阈值模型（Linear Threshold Model）。

    实现了经典的线性阈值传播模型，该模型是一种离散时间传播模型，
    每个节点都有一个随机阈值，当其已激活邻居的影响力总和超过该阈值时，
    节点被激活。每个节点对其他节点的影响力通过边权重表示。

    Attributes:
        activated_nodes (set): 所有已被激活的节点集合
        graph (IMGraph): 表示传播网络结构（继承自BaseDiffusionModel）
        init_seeds (list): 初始种子集（继承自BaseDiffusionModel）
        record_states (bool): 指示是否记录传播过程中的状态变化（继承自BaseDiffusionModel）
        states (list): 当record_states为True时存储传播过程的状态历史（继承自BaseDiffusionModel）
    """

    def __init__(self, graph: IMGraph, init_seeds: list, record_states: bool = False):
        """
        初始化线性阈值模型。

        Args:
            graph (IMGraph): 定义了节点和边的网络结构
            init_seeds (list): 初始激活的节点集合
            record_states (bool): 控制是否记录每一步的状态，默认为False
        """
        super(LinearThresholdModel, self).__init__(graph, init_seeds, record_states)
        self.activated_nodes = set(self.init_seeds)

    def update(self, current_activated_nodes: set):
        """
        执行一次传播更新。

        在线性阈值模型中，未激活的节点检查其已激活邻居的影响力总和，
        如果超过节点的阈值则被激活。

        Args:
            current_activated_nodes (set): 当前轮次需要尝试传播的已激活节点集合

        Returns:
            set: 本轮新激活的节点集合
        """
        # 执行一次传播过程
        new_activated_nodes = set()
        node_theta = {node: random.random() for node in range(self.graph.number_of_nodes)}
        for node in current_activated_nodes:
            for neighbor in self.graph.neighbors(node):
                if neighbor not in self.activated_nodes:
                    in_neighbors = self.graph.in_neighbors(neighbor)
                    sum_weights = sum(self.graph.edges[in_neighbor, neighbor]['weight']
                                      for in_neighbor in in_neighbors
                                      if in_neighbor in current_activated_nodes)
                    if sum_weights >= node_theta[neighbor]:
                        new_activated_nodes.add(neighbor)
                        self.activated_nodes.add(neighbor)
        # 记录每轮的状态
        if self.record_states and new_activated_nodes:
            self.states.append(new_activated_nodes.copy())

        return new_activated_nodes

    def diffusion(self, update_counts: int = None):
        """
        执行完整的扩散过程。

        进行指定轮次的传播更新，直到达到最大轮次或没有新的节点被激活。

        Args:
            update_counts (int, optional): 最大更新轮次数。如果为None则持续传播直到无新节点激活

        Returns:
            set: 最终所有被激活的节点集合
        """
        count = 0
        current_activated_nodes = set(self.init_seeds)
        while True:
            current_activated_nodes = self.update(current_activated_nodes)
            count += 1
            if not current_activated_nodes or (update_counts and count >= update_counts):
                break
        return self.activated_nodes

    def run_monte_carlo_diffusion(self, round: int, multi_process: bool = False, processes: int = None, seed: int = None):
        """
        执行蒙特卡洛模拟扩散过程。

        Args:
            round (int): 总模拟轮数
            multi_process (bool): 是否启用多进程模式，默认为False
            processes (int, optional): 多进程模式下的进程数，为None时使用CPU核心数
            seed (int, optional): 模拟时的随机种子

        Returns:
            float: 所有模拟轮次的平均激活节点数
        """
        return run_monte_carlo_diffusion(self, round, multi_process, processes, seed)

    def reset(self, init_seeds=None):
        """
        重置模型状态。

        将模型恢复到初始状态，如果启用状态记录则清空已记录的状态历史，
        并重新设置初始种子节点状态。

        Args:
            init_seeds (list, optional): 新的初始种子节点集合，若为None则使用原有种子集
        """
        if init_seeds is None:
            init_seeds = self.init_seeds
        self.init_seeds = init_seeds
        self.activated_nodes = set(init_seeds)
        if self.record_states:
            self.states = [set(self.init_seeds)]
