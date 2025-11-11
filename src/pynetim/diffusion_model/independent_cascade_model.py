import random

from ..graph import IMGraph
from .base_diffusion_model import BaseDiffusionModel
from .run_monte_carlo_diffusion import run_monte_carlo_diffusion


class IndependentCascadeModel(BaseDiffusionModel):
    """
    独立级联模型（Independent Cascade Model）。

    实现了经典的独立级联传播模型，该模型是一种离散时间传播模型，
    在每一轮中，已激活的节点尝试激活其未激活的邻居节点。
    每条边都有固定的传播概率，当节点尝试传播时，根据概率决定是否成功激活邻居。

    Attributes:
        activated_nodes (set): 所有已被激活的节点集合
        graph (IMGraph): 表示传播网络结构（继承自BaseDiffusionModel）
        init_seeds (list): 初始种子集（继承自BaseDiffusionModel）
        record_states (bool): 指示是否记录传播过程中的状态变化（继承自BaseDiffusionModel）
        states (list): 当record_states为True时存储传播过程的状态历史（继承自BaseDiffusionModel）
    """

    def __init__(self, graph: IMGraph, init_seeds: list, record_states: bool = False):
        """
        初始化独立级联模型。

        Args:
            graph (IMGraph): 定义了节点和边的网络结构
            init_seeds (list): 初始激活的节点集合
            record_states (bool): 控制是否记录每一步的状态，默认为False
        """
        super(IndependentCascadeModel, self).__init__(graph, init_seeds, record_states)
        self.activated_nodes = set(init_seeds)

    def update(self, current_activated_nodes: set):
        """
        执行一次传播更新。

        在当前轮次中，所有新激活的节点尝试激活它们的未激活邻居节点。
        成功激活的节点将在下一轮继续传播。

        Args:
            current_activated_nodes (set): 当前轮次需要尝试传播的已激活节点集合

        Returns:
            set: 本轮新激活的节点集合
        """
        # 执行一次传播过程
        new_activated_nodes = set()
        for node in current_activated_nodes:
            for neighbor in self.graph.neighbors(node):
                if neighbor not in self.activated_nodes:
                    if self.graph.edges[node, neighbor]['weight'] > random.random():
                        new_activated_nodes.add(neighbor)
                        self.activated_nodes.add(neighbor)

        # 记录每轮的状态（应该记录所有已激活的节点）
        if self.record_states and new_activated_nodes:
            self.states.append(new_activated_nodes.copy())

        return new_activated_nodes

    def diffusion(self, update_counts=None):
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

    def run_monte_carlo_diffusion(self, round: int, multi_process: bool = False, processes: int = None):
        """
        执行蒙特卡洛模拟扩散过程。

        Args:
            round (int): 总模拟轮数
            multi_process (bool): 是否启用多进程模式，默认为False
            processes (int, optional): 多进程模式下的进程数，为None时使用CPU核心数

        Returns:
            float: 所有模拟轮次的平均激活节点数
        """
        return run_monte_carlo_diffusion(self, round, multi_process, processes)

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
