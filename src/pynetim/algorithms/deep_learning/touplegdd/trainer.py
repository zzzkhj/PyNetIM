import os
import random
from collections import deque, namedtuple
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_max
from tqdm import tqdm

from . import models
from .environment import IMEnvironment

if TYPE_CHECKING:
    from ...graph import IMGraph

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'graph'))


class ReplayMemory:
    """经验回放缓冲区。

    存储和采样训练经验。
    """

    def __init__(self, capacity: int):
        """初始化缓冲区。

        Args:
            capacity: 缓冲区最大容量。
        """
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        """添加一条经验。"""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """随机采样一批经验。

        Args:
            batch_size: 批量大小。

        Returns:
            List[Transition]: 采样的经验列表。
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class _BaseTrainer:
    """深度强化学习训练器基类。

    提供 ToupleGDD 和 S2V-DQN 训练的公共逻辑。

    Attributes:
        model: 神经网络模型。
        device: 计算设备。
        gamma: 折扣因子。
        n_step: n-step Q-learning 的步数。
        memory: 经验回放缓冲区。
    """

    def __init__(
        self,
        model_type: str,
        device: str = 'auto',
        gamma: float = 0.99,
        n_step: int = 1,
        memory_size: int = 10000,
        batch_size: int = 64,
        lr: float = 1e-4,
        double_dqn: bool = True,
        embed_dim: int = 50,
        reg_hidden: int = 32,
        T: int = 3
    ):
        """初始化训练器。

        Args:
            model_type: 模型类型，支持 'Tripling' 或 'S2V_DQN'。
            device: 计算设备，支持 'auto'、'cpu'、'cuda'。
            gamma: 折扣因子，默认为 0.99。
            n_step: n-step Q-learning 的步数，默认为 1。
            memory_size: 经验回放缓冲区大小，默认为 10000。
            batch_size: 批量大小，默认为 64。
            lr: 学习率，默认为 1e-4。
            double_dqn: 是否使用 Double DQN，默认为 True。
            embed_dim: 嵌入维度，默认为 50。
            reg_hidden: 回归隐藏层维度，默认为 32。
            T: GNN 层数，默认为 3。
        """
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device)
        self.model_type = model_type
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.embed_dim = embed_dim
        self.reg_hidden = reg_hidden
        self.T = T

        self.memory = ReplayMemory(memory_size)
        self._node_embed_cache = {}

        self._init_model(lr)

    def _init_model(self, lr: float):
        """初始化模型和优化器。

        Args:
            lr: 学习率。
        """
        w_scale = 0.01

        if self.model_type == 'Tripling':
            self.model = models.Tripling(
                embed_dim=self.embed_dim,
                sgate_l1_dim=128,
                tgate_l1_dim=128,
                T=self.T,
                hidden_dims=[50, 50, 50],
                w_scale=w_scale
            ).to(self.device)
            if self.double_dqn:
                self.target_model = models.Tripling(
                    embed_dim=self.embed_dim,
                    sgate_l1_dim=128,
                    tgate_l1_dim=128,
                    T=self.T,
                    hidden_dims=[50, 50, 50],
                    w_scale=w_scale
                ).to(self.device)
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_model.eval()

        elif self.model_type == 'S2V_DQN':
            self.model = models.S2V_DQN(
                reg_hidden=self.reg_hidden,
                embed_dim=64,
                node_dim=2,
                edge_dim=4,
                T=self.T,
                w_scale=w_scale,
                avg=False
            ).to(self.device)
            if self.double_dqn:
                self.target_model = models.S2V_DQN(
                    reg_hidden=self.reg_hidden,
                    embed_dim=64,
                    node_dim=2,
                    edge_dim=4,
                    T=self.T,
                    w_scale=w_scale,
                    avg=False
                ).to(self.device)
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_model.eval()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def _get_node_embed(self, graph: 'IMGraph', num_epochs: int = 30) -> torch.Tensor:
        """获取节点嵌入。

        Args:
            graph: 图对象。
            num_epochs: DeepWalk 训练轮数。

        Returns:
            torch.Tensor: 节点嵌入矩阵。
        """
        graph_id = id(graph)
        if graph_id not in self._node_embed_cache:
            self._node_embed_cache[graph_id] = models.get_init_node_embed(graph, num_epochs, self.device)
        return self._node_embed_cache[graph_id]

    def _setup_graph_input(self, graph: 'IMGraph', state: torch.Tensor, action: Optional[torch.Tensor] = None) -> Data:
        """构建图输入数据。

        Args:
            graph: 图对象。
            state: 状态向量。
            action: 动作索引（可选）。

        Returns:
            Data: PyG 数据对象。
        """
        if self.model_type == 'Tripling':
            node_embed = self._get_node_embed(graph)
            x = torch.cat((node_embed, state.detach().clone().unsqueeze(dim=1)), dim=-1)
            edges = list(graph.edges.keys())
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor([graph.edges[e] for e in edges], dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=action)

        else:
            node_dim = 2
            edge_dim = 4
            x = torch.ones(graph.num_nodes, node_dim)
            x[:, 1] = 1 - state
            edges = list(graph.edges.keys())
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.ones(graph.num_edges, edge_dim)
            edge_weights = torch.tensor([graph.edges[e] for e in edges], dtype=torch.float)
            edge_attr[:, 1] = edge_weights
            edge_attr[:, 0] = state[edge_index[0]]
            edge_attr[:, 2] = torch.abs(state[edge_index[0]] - state[edge_index[1]])
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=action)

    def _create_batch(self, graphs: List['IMGraph'], states: List[torch.Tensor],
                      actions: Optional[List[torch.Tensor]] = None) -> Data:
        """创建批量数据。

        Args:
            graphs: 图对象列表。
            states: 状态向量列表。
            actions: 动作索引列表（可选）。

        Returns:
            Data: 批量数据对象。
        """
        data_list = []
        for i, graph in enumerate(graphs):
            action = actions[i] if actions is not None else None
            data = self._setup_graph_input(graph, states[i], action)
            data_list.append(data)

        loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
        for batch in loader:
            if actions is not None:
                total_num = 0
                for i in range(1, len(data_list)):
                    total_num += data_list[i - 1].num_nodes
                    if hasattr(batch[i], 'y') and batch[i].y is not None:
                        batch[i].y = batch[i].y + total_num
            return batch.to(self.device)

    @torch.no_grad()
    def select_action(self, graph: 'IMGraph', state: torch.Tensor, epsilon: float) -> int:
        """选择动作。

        Args:
            graph: 图对象。
            state: 状态向量。
            epsilon: 探索率。

        Returns:
            int: 选择的节点索引。
        """
        available = (state == 0).nonzero().squeeze(-1).tolist()
        if not isinstance(available, list):
            available = [available]

        if random.random() < epsilon:
            return random.choice(available)

        batch = self._create_batch([graph], [state.unsqueeze(0) if state.dim() == 1 else state])
        q_values = self.model(batch).squeeze()

        q_values[state == 1] = -1e10
        return torch.argmax(q_values).item()

    def memorize(self, env: IMEnvironment):
        """将轨迹存入经验回放缓冲区。

        Args:
            env: 环境对象。
        """
        sum_rewards = [0.0]
        for reward in reversed(env.rewards):
            reward /= env.graph.num_nodes
            sum_rewards.append(reward + self.gamma * sum_rewards[-1])
        sum_rewards = sum_rewards[::-1]

        for i in range(len(env.states)):
            if i + self.n_step < len(env.states):
                self.memory.push(
                    torch.tensor(env.states[i], dtype=torch.long),
                    torch.tensor([env.actions[i]], dtype=torch.long),
                    torch.tensor(env.states[i + self.n_step], dtype=torch.long),
                    torch.tensor([sum_rewards[i] - (self.gamma ** self.n_step) * sum_rewards[i + self.n_step]],
                                 dtype=torch.float),
                    env.graph
                )
            elif i + self.n_step == len(env.states):
                self.memory.push(
                    torch.tensor(env.states[i], dtype=torch.long),
                    torch.tensor([env.actions[i]], dtype=torch.long),
                    None,
                    torch.tensor([sum_rewards[i]], dtype=torch.float),
                    env.graph
                )

    def fit(self):
        """执行一步训练。"""
        sample_size = min(self.batch_size, len(self.memory))
        transitions = self.memory.sample(sample_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            dtype=torch.bool, device=self.device
        )

        non_final_next_states = [s for s in batch.next_state if s is not None]
        non_final_graphs = [batch.graph[i] for i, s in enumerate(batch.next_state) if s is not None]

        state_batch = list(batch.state)
        action_batch = list(batch.action)
        reward_batch = torch.cat(batch.reward)
        graph_batch = list(batch.graph)

        state_action_values = self.model(
            self._create_batch(graph_batch, state_batch, action_batch)
        ).squeeze(dim=1)

        next_state_values = torch.zeros(sample_size, device=self.device)

        if len(non_final_next_states) > 0:
            batch_non_final = self._create_batch(non_final_graphs, non_final_next_states)
            target_model = self.target_model if self.double_dqn else self.model
            next_state_values[non_final_mask] = scatter_max(
                target_model(batch_non_final).squeeze(dim=1).add_(
                    torch.cat(non_final_next_states).to(self.device) * (-1e5)
                ),
                batch_non_final.batch
            )[0].clamp_(min=0).detach()

        expected_state_action_values = next_state_values * self.gamma ** self.n_step + reward_batch.to(self.device)

        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        """更新目标网络。"""
        if self.double_dqn:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path: str):
        """保存模型。

        Args:
            path: 保存路径。
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """加载模型。

        Args:
            path: 模型路径。
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        if self.double_dqn:
            self.target_model.load_state_dict(self.model.state_dict())

    def train(
        self,
        graphs: List['IMGraph'],
        budget: int,
        num_epochs: int,
        save_path: str,
        eval_graphs: Optional[List['IMGraph']] = None,
        eval_interval: int = 10,
        save_interval: int = 10,
        target_update_interval: int = 100,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 10000,
        pretrain_episodes: int = 1000,
        episodes_per_epoch: int = 10,
        verbose: bool = True
    ):
        """训练模型。

        Args:
            graphs: 训练图列表。
            budget: 种子节点预算。
            num_epochs: 训练轮数。
            save_path: 模型保存路径。
            eval_graphs: 评估图列表（可选）。
            eval_interval: 评估间隔，默认为 10。
            save_interval: 保存间隔，默认为 10。
            target_update_interval: 目标网络更新间隔，默认为 100。
            eps_start: 初始探索率，默认为 1.0。
            eps_end: 最终探索率，默认为 0.05。
            eps_decay_steps: 探索率衰减步数，默认为 10000。
            pretrain_episodes: 预训练回合数，默认为 1000。
            episodes_per_epoch: 每轮训练回合数，默认为 10。
            verbose: 是否显示进度，默认为 True。
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        if verbose:
            tqdm.write('预训练阶段...')

        for _ in range(pretrain_episodes):
            graph = random.choice(graphs)
            env = IMEnvironment(graph, budget, method='RR', num_trials=1000)
            env.reset()
            state = torch.tensor(env.state, dtype=torch.long)

            for _ in range(budget):
                action = self.select_action(graph, state, epsilon=1.0)
                _, done = env.step(action)
                state = torch.tensor(env.state, dtype=torch.long)
                if done:
                    self.memorize(env)
                    break

        if verbose:
            tqdm.write('开始训练...')

        progress = tqdm(total=num_epochs, disable=not verbose)
        total_steps = 0

        for epoch in range(num_epochs):
            eps = eps_end + max(0.0, (eps_start - eps_end) * (eps_decay_steps - total_steps) / eps_decay_steps)

            for _ in range(episodes_per_epoch):
                graph = random.choice(graphs)
                env = IMEnvironment(graph, budget, method='RR', num_trials=1000)
                env.reset()
                state = torch.tensor(env.state, dtype=torch.long)

                for _ in range(budget):
                    action = self.select_action(graph, state, epsilon=eps)
                    _, done = env.step(action)
                    state = torch.tensor(env.state, dtype=torch.long)
                    total_steps += 1
                    if done:
                        self.memorize(env)
                        break

            self.fit()

            if (epoch + 1) % eval_interval == 0 and eval_graphs:
                self.model.eval()
                total_reward = 0
                for eval_graph in eval_graphs:
                    env = IMEnvironment(eval_graph, budget, method='RR', num_trials=10000)
                    env.reset()
                    state = torch.tensor(env.state, dtype=torch.long)

                    for _ in range(budget):
                        action = self.select_action(eval_graph, state, epsilon=0.0)
                        _, done = env.step(action)
                        state = torch.tensor(env.state, dtype=torch.long)
                        if done:
                            break

                    total_reward += env.prev_inf

                avg_reward = total_reward / len(eval_graphs)
                if verbose:
                    tqdm.write(f'Epoch {epoch + 1}/{num_epochs}: Avg Reward = {avg_reward:.2f}')
                self.model.train()

            if (epoch + 1) % save_interval == 0:
                self.save(f'{save_path}.epoch{epoch + 1}')

            if (epoch + 1) % target_update_interval == 0:
                self.update_target()

            progress.update(1)

        progress.close()
        self.save(save_path)
        if verbose:
            tqdm.write(f'训练完成，模型已保存至 {save_path}')


class ToupleGDDTrainer(_BaseTrainer):
    """ToupleGDD 深度强化学习训练器。

    用于训练 ToupleGDD 模型（三重门控图神经网络 + DQN）。

    论文: ToupleGDD: A Fine-Designed Solution of Influence Maximization by Deep Reinforcement Learning
    会议: IEEE TCSS 2024

    Attributes:
        model: 神经网络模型。
        device: 计算设备。
        gamma: 折扣因子。
        n_step: n-step Q-learning 的步数。
        memory: 经验回放缓冲区。
    """

    def __init__(
        self,
        device: str = 'auto',
        gamma: float = 0.99,
        n_step: int = 1,
        memory_size: int = 10000,
        batch_size: int = 64,
        lr: float = 1e-4,
        double_dqn: bool = True,
        embed_dim: int = 50,
        T: int = 3
    ):
        """初始化 ToupleGDD 训练器。

        Args:
            device: 计算设备，支持 'auto'、'cpu'、'cuda'。
            gamma: 折扣因子，默认为 0.99。
            n_step: n-step Q-learning 的步数，默认为 1。
            memory_size: 经验回放缓冲区大小，默认为 10000。
            batch_size: 批量大小，默认为 64。
            lr: 学习率，默认为 1e-4。
            double_dqn: 是否使用 Double DQN，默认为 True。
            embed_dim: 嵌入维度，默认为 50。
            T: GNN 层数，默认为 3。
        """
        super().__init__(
            model_type='Tripling',
            device=device,
            gamma=gamma,
            n_step=n_step,
            memory_size=memory_size,
            batch_size=batch_size,
            lr=lr,
            double_dqn=double_dqn,
            embed_dim=embed_dim,
            T=T
        )


class S2VDQNTrainer(_BaseTrainer):
    """S2V-DQN 深度强化学习训练器。

    用于训练 S2V-DQN 模型（Structure2Vec + DQN）。

    论文: Learning Combinatorial Optimization Algorithms over Graphs
    会议: NeurIPS 2017

    Attributes:
        model: 神经网络模型。
        device: 计算设备。
        gamma: 折扣因子。
        n_step: n-step Q-learning 的步数。
        memory: 经验回放缓冲区。
    """

    def __init__(
        self,
        device: str = 'auto',
        gamma: float = 0.99,
        n_step: int = 1,
        memory_size: int = 10000,
        batch_size: int = 64,
        lr: float = 1e-4,
        double_dqn: bool = True,
        reg_hidden: int = 32,
        T: int = 3
    ):
        """初始化 S2V-DQN 训练器。

        Args:
            device: 计算设备，支持 'auto'、'cpu'、'cuda'。
            gamma: 折扣因子，默认为 0.99。
            n_step: n-step Q-learning 的步数，默认为 1。
            memory_size: 经验回放缓冲区大小，默认为 10000。
            batch_size: 批量大小，默认为 64。
            lr: 学习率，默认为 1e-4。
            double_dqn: 是否使用 Double DQN，默认为 True。
            reg_hidden: 回归隐藏层维度，默认为 32。
            T: GNN 层数，默认为 3。
        """
        super().__init__(
            model_type='S2V_DQN',
            device=device,
            gamma=gamma,
            n_step=n_step,
            memory_size=memory_size,
            batch_size=batch_size,
            lr=lr,
            double_dqn=double_dqn,
            reg_hidden=reg_hidden,
            T=T
        )
