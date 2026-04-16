import os
from pathlib import Path
from typing import List, Optional, Set, TYPE_CHECKING

import torch

from ..base_dl import BaseDLAlgorithm
from .agent import Agent, StudentAgent, get_q_net_input
from .trainer import BiGDNTrainer
from .models import QValueNet

if TYPE_CHECKING:
    from ....graph import IMGraph


class BiGDNAlgorithm(BaseDLAlgorithm):
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
        gamma: float = 0.99,
        epsilon: float = 1.0,
        lr: float = 0.001,
        target_update: int = 100,
        n_steps: int = 2,
        ntype: str = 'DQN',
        device: str = None,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        encoder_path: Optional[str] = None,
        verbose: bool = False
    ):
        super().__init__(graph)

        self.verbose = verbose
        self.num_features = num_features
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.target_update = target_update
        self.n_steps = n_steps
        self.ntype = ntype

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.agent = Agent(
            num_features=num_features,
            gamma=gamma,
            epsilon=epsilon,
            lr=lr,
            device=self.device,
            target_update=target_update,
            n_steps=n_steps,
            ntype=ntype,
            encoder_param_path=encoder_path
        )

        if weights_path is not None:
            self._load_weights_from_path(weights_path)
        elif pretrained:
            self._load_weights()

    def run(self, k: int, use_topk: bool = True) -> Set[int]:
        """运行算法选择种子节点。

        Args:
            k: 种子节点数量。
            use_topk: 是否使用 top-k 选择策略。
                - True: 每步选择 Q 值最大的节点。
                - False: 使用迭代选择策略。

        Returns:
            Set[int]: 选择的种子节点集合。
        """
        self.agent.q_net.eval()
        seeds = set()

        with torch.no_grad():
            for _ in range(k):
                state = [1 if i in seeds else 0 for i in range(self.graph.num_nodes)]
                state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
                data = get_q_net_input([self.graph], self.num_features, self.device)

                q_values = self.agent.q_net(data.x, data.edge_index, data.edge_weight, data.batch, state_tensor)

                selectable = [i for i in range(self.graph.num_nodes) if i not in seeds]
                if use_topk:
                    q_selectable = q_values[selectable]
                    max_idx = q_selectable.argmax().item()
                    best_node = selectable[max_idx]
                else:
                    q_selectable = q_values[selectable]
                    sorted_indices = q_selectable.argsort(descending=True)
                    for idx in sorted_indices:
                        candidate = selectable[idx.item()]
                        if candidate not in seeds:
                            best_node = candidate
                            break

                seeds.add(best_node)

                if self.verbose:
                    print(f"Selected node {best_node}, Q-value: {q_values[best_node].item():.4f}")

        return seeds

    def train(
        self,
        graphs: List['IMGraph'],
        budget: int,
        num_epochs: int = 100,
        save_path: Optional[str] = None,
        eval_graphs: Optional[List['IMGraph']] = None,
        eval_interval: int = 10,
        save_interval: int = 100,
        target_update_interval: int = 100,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 10000,
        pretrain_episodes: int = 100,
        episodes_per_epoch: int = 10,
        method: str = 'MC',
        num_trials: int = 1000,
        memory_size: int = 50000,
        batch_size: int = 16,
        encoder_path: Optional[str] = None,
        verbose: bool = True
    ) -> None:
        """训练 BiGDN 模型。

        Args:
            graphs: 训练图列表。
            budget: 种子节点预算。
            num_epochs: 训练轮数，默认为 100。
            save_path: 模型保存路径。
            eval_graphs: 评估图列表（可选）。
            eval_interval: 评估间隔，默认为 10。
            save_interval: 保存间隔，默认为 100。
            target_update_interval: 目标网络更新间隔，默认为 100。
            eps_start: 初始探索率，默认为 1.0。
            eps_end: 最终探索率，默认为 0.05。
            eps_decay_steps: 探索率衰减步数，默认为 10000。
            pretrain_episodes: 预训练回合数，默认为 100。
            episodes_per_epoch: 每轮训练回合数，默认为 10。
            method: 影响力估计方法，默认为 'MC'。
            num_trials: 影响力估计试验次数，默认为 1000。
            memory_size: 经验回放缓冲区大小，默认为 50000。
            batch_size: 批量大小，默认为 16。
            encoder_path: NodeEncoder 预训练权重路径。如果为 None，则使用默认路径。
            verbose: 是否显示进度，默认为 True。
        """
        if save_path is None:
            save_path = str(Path(__file__).parent / "weights" / "bigdn_trained.pth")

        if encoder_path is None:
            encoder_path = str(Path(__file__).parent / "weights" / "node_encoder.pth")

        trainer = BiGDNTrainer(
            num_features=self.num_features,
            gamma=self.gamma,
            lr=self.lr,
            target_update=self.target_update,
            n_steps=self.n_steps,
            ntype=self.ntype,
            memory_size=memory_size,
            batch_size=batch_size,
            device=str(self.device),
            is_student=False,
            encoder_path=encoder_path
        )

        trainer.train(
            graphs=graphs,
            budget=budget,
            num_epochs=num_epochs,
            save_path=save_path,
            eval_graphs=eval_graphs,
            eval_interval=eval_interval,
            save_interval=save_interval,
            target_update_interval=target_update_interval,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay_steps=eps_decay_steps,
            pretrain_episodes=pretrain_episodes,
            episodes_per_epoch=episodes_per_epoch,
            method=method,
            num_trials=num_trials,
            verbose=verbose
        )

        self.agent = trainer.agent

    def _get_weights_path(self) -> Path:
        """获取权重文件路径。"""
        return Path(__file__).parent / "weights" / self._weights_filename

    def _load_weights(self) -> bool:
        """加载预训练权重。"""
        weights_path = self._get_weights_path()
        if weights_path.exists():
            self.agent.q_net.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
            self.agent.target_q_net.load_state_dict(self.agent.q_net.state_dict())
            if self.verbose:
                print(f"Loaded weights from {weights_path}")
            return True
        return False

    def _load_weights_from_path(self, weights_path: str) -> bool:
        """从指定路径加载权重。

        Args:
            weights_path: 权重文件路径。

        Returns:
            bool: 是否成功加载。
        """
        path = Path(weights_path)
        if path.exists():
            self.agent.q_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.agent.target_q_net.load_state_dict(self.agent.q_net.state_dict())
            if self.verbose:
                print(f"Loaded weights from {path}")
            return True
        return False

    def save_weights(self, path: str = None) -> None:
        """保存模型权重。

        Args:
            path: 保存路径，默认为权重目录下的默认文件名。
        """
        if path is None:
            path = self._get_weights_path()
        torch.save(self.agent.q_net.state_dict(), path)
        if self.verbose:
            print(f"Saved weights to {path}")


class BiGDNSAlgorithm(BaseDLAlgorithm):
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
        gamma: float = 0.99,
        epsilon: float = 1.0,
        lr: float = 0.001,
        target_update: int = 100,
        n_steps: int = 2,
        ntype: str = 'DQN',
        alpha: float = 0.5,
        device: str = None,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        teacher_path: Optional[str] = None,
        load_teacher: bool = True,
        verbose: bool = False
    ):
        super().__init__(graph)

        self.verbose = verbose
        self.num_features = num_features
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.target_update = target_update
        self.n_steps = n_steps
        self.ntype = ntype
        self.alpha = alpha

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        teacher_q_net = QValueNet(num_features).to(self.device)
        if load_teacher:
            if teacher_path is None:
                teacher_path = Path(__file__).parent / "weights" / "bigdn_weights.pth"
            if Path(teacher_path).exists():
                try:
                    teacher_q_net.load_state_dict(torch.load(teacher_path, map_location=self.device, weights_only=True))
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
            gamma=gamma,
            epsilon=epsilon,
            lr=lr,
            device=self.device,
            teacher=teacher_q_net,
            alpha=alpha,
            target_update=target_update,
            n_steps=n_steps,
            ntype=ntype
        )

        if weights_path is not None:
            self._load_weights_from_path(weights_path)
        elif pretrained:
            self._load_weights()

    def run(self, k: int, use_topk: bool = True) -> Set[int]:
        """运行算法选择种子节点。

        Args:
            k: 种子节点数量。
            use_topk: 是否使用 top-k 选择策略。

        Returns:
            Set[int]: 选择的种子节点集合。
        """
        self.agent.q_net.eval()
        seeds = set()

        with torch.no_grad():
            for _ in range(k):
                state = [1 if i in seeds else 0 for i in range(self.graph.num_nodes)]
                state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
                data = get_q_net_input([self.graph], self.num_features, self.device)

                q_values = self.agent.q_net(data.x, data.edge_index, data.edge_weight, data.batch, state_tensor)

                selectable = [i for i in range(self.graph.num_nodes) if i not in seeds]
                if use_topk:
                    q_selectable = q_values[selectable]
                    max_idx = q_selectable.argmax().item()
                    best_node = selectable[max_idx]
                else:
                    q_selectable = q_values[selectable]
                    sorted_indices = q_selectable.argsort(descending=True)
                    for idx in sorted_indices:
                        candidate = selectable[idx.item()]
                        if candidate not in seeds:
                            best_node = candidate
                            break

                seeds.add(best_node)

                if self.verbose:
                    print(f"Selected node {best_node}, Q-value: {q_values[best_node].item():.4f}")

        return seeds

    def train(
        self,
        graphs: List['IMGraph'],
        budget: int,
        num_epochs: int = 100,
        save_path: Optional[str] = None,
        eval_graphs: Optional[List['IMGraph']] = None,
        eval_interval: int = 10,
        save_interval: int = 100,
        target_update_interval: int = 100,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 10000,
        pretrain_episodes: int = 100,
        episodes_per_epoch: int = 10,
        method: str = 'MC',
        num_trials: int = 1000,
        memory_size: int = 50000,
        batch_size: int = 16,
        teacher_path: Optional[str] = None,
        verbose: bool = True
    ) -> None:
        """训练 BiGDNS 学生模型。

        Args:
            graphs: 训练图列表。
            budget: 种子节点预算。
            num_epochs: 训练轮数，默认为 100。
            save_path: 模型保存路径。
            eval_graphs: 评估图列表（可选）。
            eval_interval: 评估间隔，默认为 10。
            save_interval: 保存间隔，默认为 100。
            target_update_interval: 目标网络更新间隔，默认为 100。
            eps_start: 初始探索率，默认为 1.0。
            eps_end: 最终探索率，默认为 0.05。
            eps_decay_steps: 探索率衰减步数，默认为 10000。
            pretrain_episodes: 预训练回合数，默认为 100。
            episodes_per_epoch: 每轮训练回合数，默认为 10。
            method: 影响力估计方法，默认为 'MC'。
            num_trials: 影响力估计试验次数，默认为 1000。
            memory_size: 经验回放缓冲区大小，默认为 50000。
            batch_size: 批量大小，默认为 16。
            teacher_path: 教师模型路径。
            verbose: 是否显示进度，默认为 True。
        """
        if save_path is None:
            save_path = str(Path(__file__).parent / "weights" / "bigdns_trained.pth")

        if teacher_path is None:
            teacher_path = str(Path(__file__).parent / "weights" / "bigdn_weights.pth")

        trainer = BiGDNTrainer(
            num_features=self.num_features,
            gamma=self.gamma,
            lr=self.lr,
            target_update=self.target_update,
            n_steps=self.n_steps,
            ntype=self.ntype,
            memory_size=memory_size,
            batch_size=batch_size,
            device=str(self.device),
            is_student=True,
            teacher_path=teacher_path,
            alpha=self.alpha
        )

        trainer.train(
            graphs=graphs,
            budget=budget,
            num_epochs=num_epochs,
            save_path=save_path,
            eval_graphs=eval_graphs,
            eval_interval=eval_interval,
            save_interval=save_interval,
            target_update_interval=target_update_interval,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay_steps=eps_decay_steps,
            pretrain_episodes=pretrain_episodes,
            episodes_per_epoch=episodes_per_epoch,
            method=method,
            num_trials=num_trials,
            verbose=verbose
        )

        self.agent = trainer.agent

    def _get_weights_path(self) -> Path:
        """获取权重文件路径。"""
        return Path(__file__).parent / "weights" / self._weights_filename

    def _load_weights(self) -> bool:
        """加载预训练权重。"""
        weights_path = self._get_weights_path()
        if weights_path.exists():
            self.agent.q_net.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
            self.agent.target_q_net.load_state_dict(self.agent.q_net.state_dict())
            if self.verbose:
                print(f"Loaded weights from {weights_path}")
            return True
        return False

    def _load_weights_from_path(self, weights_path: str) -> bool:
        """从指定路径加载权重。"""
        path = Path(weights_path)
        if path.exists():
            self.agent.q_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.agent.target_q_net.load_state_dict(self.agent.q_net.state_dict())
            if self.verbose:
                print(f"Loaded weights from {path}")
            return True
        return False

    def save_weights(self, path: str = None) -> None:
        """保存模型权重。"""
        if path is None:
            path = self._get_weights_path()
        torch.save(self.agent.q_net.state_dict(), path)
        if self.verbose:
            print(f"Saved weights to {path}")
