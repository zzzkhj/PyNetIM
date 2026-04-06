from typing import Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import IMGraph

class LinearThresholdModel:
    """线性阈值模型（LT），用于观点传播模拟。
    
    每个节点从 [theta_l, theta_h) 区间均匀随机获取一个阈值，
    当其入邻居的权重和超过阈值时被激活。
    """
    
    def __init__(self, graph: 'IMGraph', seeds: Set[int], theta_l: float = 0.0, theta_h: float = 1.0, record_activated: bool = False, record_activation_frequency: bool = False) -> None:
        """构造线性阈值模型。
        
        Args:
            graph: 图对象。
            seeds: 初始种子节点集合。
            theta_l: 阈值下界，默认为 0.0。
            theta_h: 阈值上界，默认为 1.0。
            record_activated: 是否记录激活节点，默认为 False。
            record_activation_frequency: 是否记录激活频数，默认为 False。
        """
        ...
    
    def set_seeds(self, new_seeds: Set[int]) -> None:
        """更新种子节点集合。
        
        Args:
            new_seeds: 新的种子节点集合。
        """
        ...
    
    def set_record_activated(self, record: bool) -> None:
        """启用或禁用激活节点记录。
        
        Args:
            record: 是否记录激活节点。
        """
        ...
    
    def set_record_activation_frequency(self, record: bool) -> None:
        """启用或禁用激活频数记录。
        
        Args:
            record: 是否记录激活频数。
        """
        ...
    
    def run_single_simulation(self, seed: int | None = None) -> int:
        """执行单次传播模拟。
        
        Args:
            seed: 随机种子，用于结果可重现。若为 None 则使用真随机种子。
        
        Returns:
            int: 本次模拟激活的节点数。
        """
        ...
    
    def get_activated_nodes(self) -> Set[int]:
        """获取上次模拟的激活节点集合。
        
        Returns:
            Set[int]: 激活节点集合。仅在 record_activated 为 True 时有效。
        """
        ...
    
    def get_activation_frequency(self) -> list[int]:
        """获取各节点的激活频数。
        
        Returns:
            list[int]: 激活频数列表，索引 i 表示节点 i 被激活的次数。
        """
        ...
    
    def run_monte_carlo_diffusion(self, rounds: int, seed: int | None = None, use_multithread: bool = False, num_threads: int = 0) -> float:
        """运行蒙特卡洛模拟，计算平均影响力。
        
        Args:
            rounds: 模拟次数，建议 1000-10000 次。
            seed: 随机种子，用于结果可重现。若为 None 则使用真随机种子。
            use_multithread: 是否启用多线程，默认为 False。
            num_threads: 线程数，0 表示自动检测。
        
        Returns:
            float: 平均激活节点数。
        """
        ...
