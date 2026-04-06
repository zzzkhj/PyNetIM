from typing import Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import IMGraph

class SusceptibleInfectedRecoveredModel:
    """SIR 疫情传播模型，用于模拟易感-感染-恢复传播过程。
    
    易感节点以 beta 概率被感染的邻居感染，感染节点以 gamma 概率恢复。
    """
    
    def __init__(self, graph: 'IMGraph', seeds: Set[int], beta: float, gamma: float, record_activated: bool = False, record_activation_frequency: bool = False) -> None:
        """构造 SIR 模型。
        
        Args:
            graph: 图对象。
            seeds: 初始感染节点集合。
            beta: 感染概率，必须在 (0, 1] 范围内。
            gamma: 恢复概率，必须在 (0, 1] 范围内。
            record_activated: 是否记录感染/恢复节点，默认为 False。
            record_activation_frequency: 是否记录感染/恢复频数，默认为 False。
        """
        ...
    
    def set_seeds(self, new_seeds: Set[int]) -> None:
        """更新感染节点集合。
        
        Args:
            new_seeds: 新的感染节点集合。
        """
        ...
    
    def set_beta(self, beta: float) -> None:
        """设置感染概率。
        
        Args:
            beta: 感染概率。
        """
        ...
    
    def set_gamma(self, gamma: float) -> None:
        """设置恢复概率。
        
        Args:
            gamma: 恢复概率。
        """
        ...
    
    def set_record_activated(self, record: bool) -> None:
        """启用或禁用感染/恢复节点记录。
        
        Args:
            record: 是否记录感染/恢复节点。
        """
        ...
    
    def set_record_activation_frequency(self, record: bool) -> None:
        """启用或禁用感染/恢复频数记录。
        
        Args:
            record: 是否记录感染/恢复频数。
        """
        ...
    
    def run_single_simulation(self, seed: int | None = None) -> int:
        """执行单次传播模拟。
        
        Args:
            seed: 随机种子，用于结果可重现。若为 None 则使用真随机种子。
        
        Returns:
            int: 本次模拟感染+恢复的节点数。
        """
        ...
    
    def get_activated_nodes(self) -> Set[int]:
        """获取上次模拟的感染+恢复节点集合。
        
        Returns:
            Set[int]: 感染+恢复节点集合。仅在 record_activated 为 True 时有效。
        """
        ...
    
    def get_activation_frequency(self) -> list[int]:
        """获取各节点的感染/恢复频数。
        
        Returns:
            list[int]: 感染/恢复频数列表，索引 i 表示节点 i 被感染或恢复的次数。
        """
        ...
    
    def run_monte_carlo_diffusion(self, rounds: int, seed: int | None = None, use_multithread: bool = False, num_threads: int = 0) -> float:
        """运行蒙特卡洛模拟，计算平均感染+恢复人数。
        
        Args:
            rounds: 模拟次数，建议 1000-10000 次。
            seed: 随机种子，用于结果可重现。若为 None 则使用真随机种子。
            use_multithread: 是否启用多线程，默认为 False。
            num_threads: 线程数，0 表示自动检测。
        
        Returns:
            float: 平均感染+恢复节点数。
        """
        ...
