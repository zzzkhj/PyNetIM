import random

from .run_monte_carlo_diffusion import run_monte_carlo_diffusion
from ..graph import IMGraph
from .base_diffusion_model import BaseDiffusionModel
from ..utils import infection_threshold


class SusceptibleInfectedRecoveredModel(BaseDiffusionModel):
    """
    SIR模型（Susceptible-Infected-Recovered Model）。

    实现了经典的SIR传播模型，该模型是一种传染病传播模型，
    节点有三种状态：易感(S)、感染(I)和康复(R)。
    易感节点可被感染节点感染，感染节点可以康复为免疫状态。

    Attributes:
        infected_nodes (set): 所有已被感染的节点集合
        recovered_nodes (set): 所有已康复的节点集合
        graph (IMGraph): 表示传播网络结构（继承自BaseDiffusionModel）
        init_seeds (list): 初始感染节点集（继承自BaseDiffusionModel）
        record_states (bool): 指示是否记录传播过程中的状态变化（继承自BaseDiffusionModel）
        states (list): 当record_states为True时存储传播过程的状态历史（继承自BaseDiffusionModel）
        gamma (float): 康复率
        beta (float): 感染率
    """

    def __init__(self, graph: IMGraph, init_seeds: list, gamma: float, beta: float = None,
                 record_states: bool = False):
        """
        初始化SIR模型。

        Args:
            graph (IMGraph): 定义了节点和边的网络结构
            init_seeds (list): 初始感染的节点集合
            gamma (float): 康复率，每个时间步感染节点康复的概率
            beta (float, optional): 感染率，如果为None则使用基于图结构计算的感染阈值
            record_states (bool): 控制是否记录每一步的状态，默认为False
        """
        super(SusceptibleInfectedRecoveredModel, self).__init__(graph, init_seeds, record_states)
        self.infected_nodes = set(init_seeds)
        self.recovered_nodes = set()
        self.gamma = gamma
        if beta is None:
            self.beta = infection_threshold(self.graph.nx_graph)
        else:
            self.beta = beta

    def update(self):
        """
        执行一次传播更新。

        在当前轮次中，先处理感染节点的康复，然后已感染的节点尝试感染它们的易感邻居节点。

        Returns:
            set: 本轮新感染的节点集合
        """
        # 先执行恢复过程 - 所有感染节点都有一定概率康复
        new_recovered_nodes = set()
        for node in self.infected_nodes:
            if random.random() < self.gamma:
                new_recovered_nodes.add(node)
                self.recovered_nodes.add(node)

        # 从感染节点集合中移除已康复的节点
        self.infected_nodes -= new_recovered_nodes

        # 再执行感染过程 - 所有当前感染节点尝试感染邻居
        new_infected_nodes = set()
        for node in list(self.infected_nodes):  # 使用list复制避免在迭代时修改集合
            for neighbor in self.graph.neighbors(node):
                # 只有易感节点（既未感染也未康复）才能被感染
                if neighbor not in self.infected_nodes and neighbor not in self.recovered_nodes:
                    if random.random() < self.beta:  # 使用beta作为感染概率
                        new_infected_nodes.add(neighbor)

        self.infected_nodes.update(new_infected_nodes)
        # 记录每轮的状态
        if self.record_states and (new_infected_nodes or new_recovered_nodes):
            self.states.append({
                'newly_infected': new_infected_nodes.copy(),
                'newly_recovered': new_recovered_nodes.copy()
            })

        return new_infected_nodes

    def diffusion(self, update_counts=None):
        """
        执行完整的扩散过程。

        进行指定轮次的传播更新，直到达到最大轮次或没有新的节点被感染。

        Args:
            update_counts (int, optional): 最大更新轮次数。如果为None则持续传播直到无新节点感染

        Returns:
            dict: 包含最终各状态节点集合的字典
                - 'infected': 最终被感染的节点集合
                - 'recovered': 最终已康复的节点集合
                - 'susceptible': 最终仍处于易感状态的节点集合
        """
        if update_counts is not None and update_counts <= 0:
            raise ValueError("update_counts must be a positive integer.")

        count = 0
        while True:
            _ = self.update()
            count += 1
            # 当没有新的感染节点或者达到指定轮次时停止
            if len(self.infected_nodes) == 0 or (update_counts and count >= update_counts):
                break

        return self.recovered_nodes

    def run_monte_carlo_diffusion(self, round: int, update_counts: int = None, multi_process: bool = False,
                                  processes: int = None, seed: int = None):
        """
        执行蒙特卡洛模拟扩散过程。

        Args:
            round (int): 总模拟轮数
            multi_process (bool): 是否启用多进程模式，默认为False
            processes (int, optional): 多进程模式下的进程数，为None时使用CPU核心数
            seed (int, optional): 模拟时的随机种子

        Returns:
            float: 所有模拟轮次的平均感染节点数
        """
        return run_monte_carlo_diffusion(self, round, update_counts, multi_process, processes, seed)

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
        self.recovered_nodes = set()
        if self.record_states:
            self.states = []
