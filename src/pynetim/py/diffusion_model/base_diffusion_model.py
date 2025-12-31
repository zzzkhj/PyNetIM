from ..graph import IMGraph


class BaseDiffusionModel:
    """
    扩散模型基类。

    该类为各种扩散模型实现提供了基础框架，包含图结构、初始种子集、状态记录功能，
    以及需要子类实现的核心方法。支持在网络图上模拟信息、疾病或其他
    实体的传播过程。

    Attributes:
        graph (IMGraph): 表示传播网络结构
        init_seeds (list): 初始种子集
        record_states (bool): 指示是否记录传播过程中的状态变化
        states (list): 当record_states为True时存储传播过程的状态历史
    """

    def __init__(self, graph: IMGraph, init_seeds: list, record_states: bool = False):
        """
        初始化扩散模型基类。

        Args:
            graph (IMGraph): 定义了节点和边的网络结构
            init_seeds (list): 初始激活的节点集合
            record_states (bool): 控制是否记录每一步的状态，默认为False
        """
        self.graph = graph
        self.init_seeds = init_seeds.copy()
        self.record_states = record_states

        if self.record_states:
            self.states = [set(init_seeds)]

    def update(self):
        """
        更新模型状态的抽象方法。

        子类必须实现此方法来定义具体的更新逻辑。
        每次调用应执行一次传播步骤。

        Returns:
            本轮更新结果，具体类型由子类定义

        Raises:
            NotImplementedError: 当子类未实现此方法时抛出
        """
        raise NotImplementedError

    def diffusion(self, update_counts: int = None):
        """
        执行完整扩散过程的抽象方法。

        子类必须实现此方法来定义从开始到结束的完整传播流程。

        Args:
            update_counts (int, optional): 更新次数，控制扩散迭代的轮数

        Returns:
            扩散结果，具体类型由子类定义

        Raises:
            NotImplementedError: 当子类未实现此方法时抛出
        """
        raise NotImplementedError

    def run_monte_carlo_diffusion(self, round: int, multi_process: bool = False, processes: int = None):
        """
        执行蒙特卡洛模拟扩散过程。

        Args:
            round (int): 模拟的轮数
            multi_process (bool): 指示是否使用多进程进行模拟
            processes (int, optional): 多进程的进程数，默认为None，表示使用CPU核数

        Returns:
            模拟结果，具体类型由子类定义

        Raises:
            NotImplementedError: 当子类未实现此方法时抛出
        """
        raise NotImplementedError

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
        self.init_seeds = init_seeds.copy()
        if self.record_states:
            self.states = [set(self.init_seeds)]
