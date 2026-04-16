from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch

from .base import BiGDNBaseAlgorithm
from .agent import Agent, StudentAgent
from .trainer import BiGDNTrainer
from .models import QValueNet

if TYPE_CHECKING:
    from ....graph import IMGraph


class BiGDNAlgorithm(BiGDNBaseAlgorithm):
    """BiGDN 影响力最大化算法。

    使用双向图扩散网络进行种子节点选择。

    References:
        BiGDN: An end-to-end influence maximization framework based on deep reinforcement
        learning and graph neural networks.
        Wenlong Zhu, Kaijing Zhang, Jiahui Zhong, Chengle Hou, Jie Ji.
        Expert Systems with Applications, 270:126384, 2025.
    """

    _weights_filename = "bigdn_weights.pth"

    def __init__(
        self,
        graph: 'IMGraph',
        num_features: int = 64,
        device: str = 'auto',
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        encoder_path: Optional[str] = None,
        verbose: bool = False
    ):
        """初始化 BiGDN 算法。

        Args:
            graph: 输入图对象。
            num_features: 特征维度，默认为 64。
            device: 计算设备，支持 'auto'、'cpu'、'cuda'。
            pretrained: 是否使用预训练权重，默认为 True。
            weights_path: 本地权重路径，优先级高于 pretrained。
            encoder_path: NodeEncoder 预训练权重路径。
            verbose: 是否输出详细信息，默认为 False。
        """
        super().__init__(graph, num_features=num_features, verbose=verbose)

        self.device = self._get_device(device)

        self.agent = Agent(
            num_features=num_features,
            device=self.device,
            encoder_param_path=encoder_path
        )

        if weights_path is not None:
            self._load_weights_from_path(weights_path)
        elif pretrained:
            self._load_weights()


class BiGDNSAlgorithm(BiGDNBaseAlgorithm):
    """BiGDNS 学生模型影响力最大化算法。

    使用知识蒸馏训练的轻量级学生网络。

    References:
        BiGDN: An end-to-end influence maximization framework based on deep reinforcement
        learning and graph neural networks.
        Wenlong Zhu, Kaijing Zhang, Jiahui Zhong, Chengle Hou, Jie Ji.
        Expert Systems with Applications, 270:126384, 2025.
    """

    _weights_filename = "bigdns_weights.pth"

    def __init__(
        self,
        graph: 'IMGraph',
        num_features: int = 64,
        device: str = 'auto',
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        verbose: bool = False
    ):
        """初始化 BiGDNS 算法。

        Args:
            graph: 输入图对象。
            num_features: 特征维度，默认为 64。
            device: 计算设备，支持 'auto'、'cpu'、'cuda'。
            pretrained: 是否使用预训练权重，默认为 True。
            weights_path: 本地权重路径，优先级高于 pretrained。
            verbose: 是否输出详细信息，默认为 False。
        """
        super().__init__(graph, num_features=num_features, verbose=verbose)

        self.device = self._get_device(device)

        teacher_q_net = QValueNet(num_features).to(self.device)
        teacher_path = Path(__file__).parent / "weights" / "bigdn_weights.pth"
        if Path(teacher_path).exists():
            try:
                teacher_q_net.load_state_dict(
                    torch.load(teacher_path, map_location=self.device, weights_only=True)
                )
                if self.verbose:
                    print(f"Loaded teacher weights from {teacher_path}")
            except RuntimeError as e:
                if self.verbose:
                    print(f"Warning: Failed to load teacher weights (size mismatch): {e}")
        else:
            if self.verbose:
                print(f"Warning: Teacher weights not found at {teacher_path}")

        self.agent = StudentAgent(
            num_features=num_features,
            device=self.device,
            teacher=teacher_q_net
        )

        if weights_path is not None:
            self._load_weights_from_path(weights_path)
        elif pretrained:
            self._load_weights()
