import os
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import torch
from tqdm import tqdm

from .agent import Agent, StudentAgent, ReplayBuffer, get_q_net_input
from .environment import GraphEnvironment
from .models import QValueNet, StudentQValueNet, NodeEncoder

if TYPE_CHECKING:
    from ....graph import IMGraph


class BiGDNTrainer:
    """BiGDN 训练器。

    用于训练 BiGDN 和 BiGDNS 模型。

    References:
        BiGDN: An end-to-end influence maximization framework based on deep reinforcement
        learning and graph neural networks.
        Wenlong Zhu, Kaijing Zhang, Jiahui Zhong, Chengle Hou, Jie Ji.
        Expert Systems with Applications, 270:126384, 2025.
    """

    def __init__(
        self,
        num_features: int = 64,
        gamma: float = 0.99,
        lr: float = 0.001,
        target_update: int = 100,
        n_steps: int = 2,
        ntype: str = 'DQN',
        memory_size: int = 50000,
        batch_size: int = 16,
        device: str = 'auto',
        is_student: bool = False,
        teacher_path: Optional[str] = None,
        encoder_path: Optional[str] = None,
        alpha: float = 0.5
    ):
        """初始化训练器。

        Args:
            num_features: 特征维度，默认为 64。
            gamma: 折扣因子，默认为 0.99。
            lr: 学习率，默认为 0.001。
            target_update: 目标网络更新间隔，默认为 100。
            n_steps: n-step Q-learning 的步数，默认为 2。
            ntype: 网络类型，默认为 'DQN'。
            memory_size: 经验回放缓冲区大小，默认为 50000。
            batch_size: 批量大小，默认为 16。
            device: 计算设备，支持 'auto'、'cpu'、'cuda'。
            is_student: 是否为学生模型，默认为 False。
            teacher_path: 教师模型路径（仅学生模型使用）。
            encoder_path: NodeEncoder 预训练权重路径（仅 BiGDN 使用，BiGDNS 不需要）。
            alpha: 知识蒸馏权重（仅学生模型使用）。
        """
        self.num_features = num_features
        self.gamma = gamma
        self.lr = lr
        self.target_update = target_update
        self.n_steps = n_steps
        self.ntype = ntype
        self.batch_size = batch_size
        self.is_student = is_student
        self.alpha = alpha

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.memory = ReplayBuffer(memory_size)

        if is_student:
            teacher_q_net = QValueNet(num_features).to(self.device)
            if teacher_path is not None and Path(teacher_path).exists():
                teacher_q_net.load_state_dict(
                    torch.load(teacher_path, map_location=self.device, weights_only=True)
                )
            self.agent = StudentAgent(
                num_features=num_features,
                gamma=gamma,
                epsilon=1.0,
                lr=lr,
                device=self.device,
                teacher=teacher_q_net,
                alpha=alpha,
                target_update=target_update,
                n_steps=n_steps,
                ntype=ntype
            )
        else:
            self.agent = Agent(
                num_features=num_features,
                gamma=gamma,
                epsilon=1.0,
                lr=lr,
                device=self.device,
                target_update=target_update,
                n_steps=n_steps,
                ntype=ntype,
                encoder_param_path=encoder_path
            )

    def select_action(self, graph: 'IMGraph', state: List[int], epsilon: float) -> int:
        """选择动作。

        Args:
            graph: 图对象。
            state: 状态向量。
            epsilon: 探索率。

        Returns:
            int: 选择的节点索引。
        """
        import random
        available = [i for i in range(len(state)) if state[i] == 0]
        if not available:
            return -1

        if random.random() < epsilon:
            return random.choice(available)

        self.agent.q_net.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
            data = get_q_net_input([graph], self.num_features, self.device)
            q_values = self.agent.q_net(data.x, data.edge_index, data.edge_weight, data.batch, state_tensor)

            for i in range(len(state)):
                if state[i] == 1:
                    q_values[i] = -1e10

            best_node = q_values.argmax().item()
        self.agent.q_net.train()
        return best_node

    def memorize(self, env: GraphEnvironment):
        """将轨迹存入经验回放缓冲区。

        Args:
            env: 环境对象。
        """
        for i in range(len(env.actions)):
            state = env.states[i]
            action = env.actions[i]
            reward = env.rewards[i]

            if i + 1 < len(env.states):
                next_state = env.states[i + 1]
                done = (i == len(env.actions) - 1)
            else:
                next_state = [0] * len(state)
                done = True

            self.memory.add(state, action, reward, next_state, done, env.graph)

    def fit(self):
        """执行一步训练。"""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, graphs = self.memory.sample(self.batch_size)
        self.agent.update(states, actions, rewards, next_states, graphs, dones)

    def update_target(self):
        """更新目标网络。"""
        self.agent.target_q_net.load_state_dict(self.agent.q_net.state_dict())

    def save(self, path: str):
        """保存模型。

        Args:
            path: 保存路径。
        """
        torch.save(self.agent.q_net.state_dict(), path)

    def load(self, path: str):
        """加载模型。

        Args:
            path: 模型路径。
        """
        self.agent.q_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.agent.target_q_net.load_state_dict(self.agent.q_net.state_dict())

    def train(
        self,
        graphs: List['IMGraph'],
        budget: int,
        num_epochs: int,
        save_path: str,
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
            save_interval: 保存间隔，默认为 100。
            target_update_interval: 目标网络更新间隔，默认为 100。
            eps_start: 初始探索率，默认为 1.0。
            eps_end: 最终探索率，默认为 0.05。
            eps_decay_steps: 探索率衰减步数，默认为 10000。
            pretrain_episodes: 预训练回合数，默认为 100。
            episodes_per_epoch: 每轮训练回合数，默认为 10。
            method: 影响力估计方法，默认为 'MC'。
            num_trials: 影响力估计试验次数，默认为 1000。
            verbose: 是否显示进度，默认为 True。
        """
        import random

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        if verbose:
            tqdm.write('预训练阶段...')

        for _ in range(pretrain_episodes):
            graph = random.choice(graphs)
            env = GraphEnvironment([graph], k=budget, gamma=self.gamma, n_steps=self.n_steps, method=method, num_trials=num_trials)
            state = env.reset()

            for _ in range(budget):
                action = self.select_action(graph, state, epsilon=1.0)
                if action == -1:
                    break
                _, state, done = env.step(action)
                if done:
                    self.memorize(env)
                    break

        if verbose:
            tqdm.write('开始训练...')

        progress = tqdm(total=num_epochs, disable=not verbose, desc=f'使用 {len(graphs)} 个图训练 {"BiGDNS" if self.is_student else "BiGDN"}')
        total_steps = 0

        for epoch in range(num_epochs):
            eps = eps_end + max(0.0, (eps_start - eps_end) * (eps_decay_steps - total_steps) / eps_decay_steps)

            for _ in range(episodes_per_epoch):
                graph = random.choice(graphs)
                env = GraphEnvironment([graph], k=budget, gamma=self.gamma, n_steps=self.n_steps, method=method, num_trials=num_trials)
                state = env.reset()

                for _ in range(budget):
                    action = self.select_action(graph, state, epsilon=eps)
                    if action == -1:
                        break
                    _, state, done = env.step(action)
                    total_steps += 1
                    if done:
                        self.memorize(env)
                        break

            self.fit()

            if (epoch + 1) % target_update_interval == 0:
                self.update_target()

            if (epoch + 1) % eval_interval == 0 and eval_graphs:
                self.agent.q_net.eval()
                total_reward = 0
                for eval_graph in eval_graphs:
                    env = GraphEnvironment([eval_graph], k=budget, gamma=self.gamma, n_steps=self.n_steps, method=method, num_trials=num_trials * 10)
                    state = env.reset()

                    for _ in range(budget):
                        action = self.select_action(eval_graph, state, epsilon=0.0)
                        if action == -1:
                            break
                        _, state, done = env.step(action)
                        if done:
                            break

                    total_reward += env.preview_reward

                avg_reward = total_reward / len(eval_graphs)
                if verbose:
                    tqdm.write(f'Epoch {epoch + 1}/{num_epochs}: Avg Reward = {avg_reward:.2f}, Buffer = {len(self.memory)}')
                self.agent.q_net.train()

            if (epoch + 1) % save_interval == 0:
                self.save(f'{save_path}.epoch{epoch + 1}')

            progress.update(1)

        progress.close()
        self.save(save_path)
        if verbose:
            tqdm.write(f'训练完成，模型已保存至 {save_path}')


class BiGDNNodeEncoderTrainer:
    """BiGDN NodeEncoder 预训练器。

    通过监督学习预训练 NodeEncoder，预测每个节点单独作为种子节点时的
    归一化影响力值。预训练后的 NodeEncoder 可用于 BiGDN 模型初始化。

    Attributes:
        encoder: NodeEncoder 模型。
        device: 计算设备。
        num_features: 特征维度。
    """

    def __init__(
        self,
        num_features: int = 64,
        lr: float = 0.001,
        device: str = 'auto'
    ):
        """初始化 BiGDN NodeEncoder 预训练器。

        Args:
            num_features: 特征维度，默认为 64。
            lr: 学习率，默认为 0.001。
            device: 计算设备，支持 'auto'、'cpu'、'cuda'。
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.num_features = num_features
        self.encoder = NodeEncoder(num_features).to(self.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self._data_list = []

    def prepare_data(
        self,
        graphs: List['IMGraph'],
        num_trials: int = 10000,
        verbose: bool = True
    ) -> None:
        """准备训练数据。

        为每个图计算每个节点的影响力值作为标签。

        Args:
            graphs: 训练图列表。
            num_trials: MC 模拟次数，用于计算影响力值，默认为 10000。
            verbose: 是否显示进度，默认为 True。
        """
        import numpy as np
        from torch_geometric.data import Data

        self._data_list = []

        if verbose:
            tqdm.write('准备训练数据...')

        for graph in graphs:
            y_g = np.zeros(graph.num_nodes, dtype=float)
            for node in range(graph.num_nodes):
                y_g[node] = _compute_influence_mc(graph, [node], num_trials) / graph.num_nodes

            y = y_g
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)

            edge_index = [[], []]
            edge_weight = []
            for u in range(graph.num_nodes):
                for v, w in graph.out_neighbors_with_weights(u):
                    edge_index[0].append(u)
                    edge_index[1].append(v)
                    edge_weight.append(w)

            x = torch.ones(graph.num_nodes, self.num_features * 2, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            data = Data(x, edge_index, y=y, edge_weight=edge_weight)
            self._data_list.append(data.to(self.device))

    def train(
        self,
        graphs: List['IMGraph'],
        num_epochs: int = 100,
        save_path: Optional[str] = None,
        num_trials: int = 10000,
        verbose: bool = True
    ) -> NodeEncoder:
        """训练 NodeEncoder。

        Args:
            graphs: 训练图列表。
            num_epochs: 训练轮数，默认为 100。
            save_path: 模型保存路径。
            num_trials: MC 模拟次数，用于计算影响力值，默认为 10000。
            verbose: 是否显示进度，默认为 True。

        Returns:
            NodeEncoder: 训练好的 NodeEncoder。
        """
        if not self._data_list:
            self.prepare_data(graphs, num_trials, verbose)

        progress = tqdm(total=num_epochs, disable=not verbose, desc='预训练 NodeEncoder')

        for epoch in range(num_epochs):
            total_loss = 0
            for data in self._data_list:
                self.optimizer.zero_grad()
                out, _ = self.encoder(data.x, data.edge_index, data.edge_weight)
                loss = torch.nn.functional.mse_loss(out, data.y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(self._data_list)
                tqdm.write(f'Epoch {epoch + 1}/{num_epochs}: Avg Loss = {avg_loss:.4f}')

            progress.update(1)

        progress.close()

        if save_path is not None:
            self.save(save_path)
            if verbose:
                tqdm.write(f'NodeEncoder 已保存至 {save_path}')

        return self.encoder

    def save(self, path: str) -> None:
        """保存模型。

        Args:
            path: 保存路径。
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(self.encoder.state_dict(), path)

    def load(self, path: str) -> None:
        """加载模型。

        Args:
            path: 模型路径。
        """
        self.encoder.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


def _compute_influence_mc(graph: 'IMGraph', seeds: List[int], num_trials: int = 1000) -> float:
    """使用 MC 模拟计算影响力。

    Args:
        graph: 图对象。
        seeds: 种子节点列表。
        num_trials: 模拟次数。

    Returns:
        float: 平均影响力值。
    """
    if not seeds:
        return 0.0

    from ....diffusion_model import IndependentCascadeModel

    ic_model = IndependentCascadeModel(graph, set(seeds))
    avg_activated = ic_model.run_monte_carlo_diffusion(num_trials)
    return avg_activated
