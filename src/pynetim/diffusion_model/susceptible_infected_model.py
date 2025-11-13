import random

from .base_diffusion_model import BaseDiffusionModel
from .run_monte_carlo_diffusion import run_monte_carlo_diffusion
from ..graph import IMGraph
from ..utils import infection_threshold


class SusceptibleInfectedModel(BaseDiffusionModel):
    """
    SI模型（Susceptible-Infected Model）。

    实现了经典的SI传播模型，该模型是一种简单的传染病传播模型，
    节点只有两种状态：易感(S)和感染(I)。一旦节点被感染，它将始终保持感染状态
    并持续尝试感染其邻居节点。

    Attributes:
        infected_nodes (set): 所有已被感染的节点集合
        graph (IMGraph): 表示传播网络结构（继承自BaseDiffusionModel）
        init_seeds (list): 初始感染节点集（继承自BaseDiffusionModel）
        record_states (bool): 指示是否记录传播过程中的状态变化（继承自BaseDiffusionModel）
        states (list): 当record_states为True时存储传播过程的状态历史（继承自BaseDiffusionModel）
        beta (float): 感染概率
    """

    def __init__(self, graph: IMGraph, init_seeds: list, beta: float = None, record_states: bool = False):
        """
        初始化SI模型。

        Args:
            graph (IMGraph): 定义了节点和边的网络结构
            init_seeds (list): 初始感染的节点集合
            beta (float, optional): 感染概率，如果为None则使用基于图结构计算的感染阈值
            record_states (bool): 控制是否记录每一步的状态，默认为False
        """
        super(SusceptibleInfectedModel, self).__init__(graph, init_seeds, record_states)
        self.infected_nodes = set(init_seeds)
        if beta is None:
            self.beta = infection_threshold(self.graph.nx_graph)
        else:
            self.beta = beta

    def update(self):
        """
        执行一次传播更新。

        在当前轮次中，所有已感染的节点尝试感染它们的易感邻居节点。
        成功感染的节点将在后续所有轮次中继续传播。

        Returns:
            set: 本轮新感染的节点集合
        """
        # 执行一次传播过程
        new_infected_nodes = set()
        # 遍历所有已感染的节点，而不是仅当前轮次新感染的节点
        for node in self.infected_nodes:
            for neighbor in self.graph.neighbors(node):
                if neighbor not in self.infected_nodes:
                    if self.beta > random.random():
                        new_infected_nodes.add(neighbor)

        self.infected_nodes.update(new_infected_nodes)
        # 记录每轮的状态
        if self.record_states and new_infected_nodes:
            self.states.append(new_infected_nodes.copy())

        return new_infected_nodes

    def diffusion(self, update_counts):
        """
        执行完整的扩散过程。

        进行指定轮次的传播更新，直到达到最大轮次或没有新的节点被感染。

        Args:
            update_counts (int): 最大更新轮次数。

        Returns:
            set: 最终所有被感染的节点集合
        """
        if update_counts is not None and update_counts <= 0:
            raise ValueError("update_counts must be a positive integer.")
        count = 0
        while True:
            _ = self.update()
            count += 1
            if (len(self.infected_nodes) == self.graph.number_of_nodes) or (update_counts and count >= update_counts):
                break
        return self.infected_nodes

    def run_monte_carlo_diffusion(self, rounds: int, update_counts: int, multi_process: bool = False,
                                  processes: int = None, seed: int = None):
        """
        执行蒙特卡洛模拟扩散过程。

        Args:
            rounds (int): 总模拟轮数
            update_counts (int): 更新轮次数，适用于SI模型等需要限制传播轮次的模型
            multi_process (bool): 是否启用多进程模式，默认为False
            processes (int, optional): 多进程模式下的进程数，为None时使用CPU核心数
            seed (int, optional): 模拟时的随机种子

        Returns:
            float: 所有模拟轮次的平均激活节点数
        """
        return run_monte_carlo_diffusion(self, rounds, update_counts, multi_process, processes, seed)


    def reset(self, init_seeds=None):
        """
        重置模型状态。

        将模型恢复到初始状态，如果启用状态记录则清空已记录的状态历史，
        并重新设置初始感染节点状态。

        Args:
            init_seeds (list, optional): 新的初始感染节点集合，若为None则使用原有种子集
        """
        if init_seeds is None:
            init_seeds = self.init_seeds
        self.init_seeds = init_seeds
        self.infected_nodes = set(init_seeds)
        if self.record_states:
            self.states = [set(self.init_seeds)]
