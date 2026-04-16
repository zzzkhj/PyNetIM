import random
from collections import deque
from typing import List, Tuple, TYPE_CHECKING

import torch
import torch_scatter
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from .models import QValueNet, StudentQValueNet

if TYPE_CHECKING:
    from ....graph import IMGraph


class ReplayBuffer:
    """经验回放缓冲区（内部使用，不向外导出）。"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state: List[int], action: int, reward: float, next_state: List[int], done: bool, graph):
        self.buffer.append((state, action, reward, next_state, done, graph))

    def sample(self, batch_size: int) -> Tuple:
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, graph = zip(*transitions)
        return state, action, reward, next_state, done, graph

    def size(self) -> int:
        return len(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)


class Agent:
    """BiGDN Agent。

    References:
        BiGDN: An end-to-end influence maximization framework based on deep reinforcement
        learning and graph neural networks.
        Wenlong Zhu, Kaijing Zhang, Jiahui Zhong, Chengle Hou, Jie Ji.
        Expert Systems with Applications, 270:126384, 2025.
    """

    def __init__(
        self,
        num_features: int,
        device: torch.device,
        gamma: float = 0.99,
        epsilon: float = 0.0,
        lr: float = 0.001,
        target_update: int = 100,
        n_steps: int = 2,
        encoder_param_path: str = None,
        ntype: str = 'DQN',
        training: bool = False
    ):
        self.num_features = num_features
        self.q_net = QValueNet(num_features).to(device)
        self.q_net.apply(self.init_weights)

        if encoder_param_path is not None:
            self.q_net.encoder.load_state_dict(torch.load(encoder_param_path, map_location=device, weights_only=True))

        self.target_q_net = QValueNet(num_features).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.count = 0
        self.target_update = target_update
        self.n_steps = n_steps
        self.ntype = ntype
        self.training = training

    @torch.no_grad()
    def take_action(self, state: List[int], env) -> int:
        selectable_nodes = list(set(range(env.graph.num_nodes)) - set(env.seeds))
        if random.random() < self.epsilon:
            node = random.choice(selectable_nodes)
        else:
            selectable_nodes_t = torch.tensor(selectable_nodes, dtype=torch.long, device=self.device)
            states = torch.tensor(state, dtype=torch.float, device=self.device)
            data = get_q_net_input([env.graph], self.num_features, self.device)
            self.q_net.eval()
            q_values = self.q_net(data.x, data.edge_index, data.edge_weight, data.batch, states)
            q_values_selectable = q_values[selectable_nodes]
            max_q_value, _ = q_values_selectable.max(0)
            max_indices = (q_values_selectable == max_q_value).nonzero(as_tuple=False).squeeze()

            if max_indices.numel() == 1:
                max_index = max_indices.item()
            else:
                max_index = max_indices[random.randint(0, len(max_indices) - 1)].item()

            node = selectable_nodes[max_index]

        return node

    def update(self, states, actions, rewards, next_states, graphs, dones) -> float:
        states = torch.tensor([s for state in states for s in state], dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        next_states = torch.tensor(
            [s for state in next_states for s in state], dtype=torch.float, device=self.device
        ).view(-1)
        dones = torch.tensor(dones, dtype=torch.int, device=self.device)

        data = get_q_net_input(graphs, self.num_features, self.device)

        self.q_net.train()
        actions_q_values = self.q_net(data.x, data.edge_index, data.edge_weight, data.batch, states)
        bidx = torch.unique(data.batch)
        len_b = 0
        for b in bidx:
            actions[b] += len_b
            len_b += (data.batch == b).sum()

        q_values = actions_q_values.gather(dim=0, index=actions)
        self.target_q_net.eval()
        with torch.no_grad():
            if self.ntype == 'DQN':
                max_q_values = torch_scatter.scatter_max(
                    self.target_q_net(data.x, data.edge_index, data.edge_weight, data.batch, next_states)
                    + next_states * -1e6,
                    data.batch
                )[0].clamp_(min=0)
            else:
                next_q_values = self.q_net(
                    data.x, data.edge_index, data.edge_weight, data.batch, next_states
                )
                max_actions = torch_scatter.scatter_max(next_q_values + next_states * -1e6, data.batch)[1]
                max_q_values = (
                    self.target_q_net(data.x, data.edge_index, data.edge_weight, data.batch, next_states)
                    .gather(dim=0, index=max_actions)
                ).clamp_(min=0)

        q_targets = rewards + self.gamma ** self.n_steps * max_q_values * (1 - dones)
        self.optim.zero_grad()
        loss = F.mse_loss(q_values, q_targets.detach())
        loss.backward()
        self.optim.step()

        self.count += 1
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        return loss.item()

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)


class StudentAgent:
    """学生 Agent（用于知识蒸馏）。"""

    def __init__(
        self,
        num_features: int,
        device: torch.device,
        teacher,
        gamma: float = 0.99,
        epsilon: float = 0.0,
        lr: float = 0.001,
        alpha: float = 0.5,
        target_update: int = 100,
        n_steps: int = 2,
        ntype: str = 'DQN',
        training: bool = False
    ):
        self.num_features = num_features
        self.q_net = StudentQValueNet(num_features).to(device)
        self.q_net.apply(self.init_weights)

        self.target_q_net = StudentQValueNet(num_features).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.teacher = teacher
        self.alpha = alpha

        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.count = 0
        self.target_update = target_update
        self.n_steps = n_steps
        self.ntype = ntype
        self.training = training

        self.step = (0.9 - 0.1) / (1000 - 1)

    @torch.no_grad()
    def take_action(self, state: List[int], env) -> int:
        selectable_nodes = list(set(range(env.graph.num_nodes)) - set(env.seeds))
        if random.random() < self.epsilon:
            node = random.choice(selectable_nodes)
        else:
            selectable_nodes_t = torch.tensor(selectable_nodes, dtype=torch.long, device=self.device)
            states = torch.tensor(state, dtype=torch.float, device=self.device)
            data = get_q_net_input([env.graph], self.num_features, self.device)
            self.q_net.eval()
            q_values = self.q_net(data.x, data.edge_index, data.edge_weight, data.batch, states)
            selectable_q_values_sort = q_values[selectable_nodes_t].sort(descending=True).values
            for mq in selectable_q_values_sort:
                max_position = set((q_values == mq).nonzero().view(-1).tolist())
                nodes = list(set(selectable_nodes).intersection(max_position))
                if len(nodes) > 0:
                    node = random.choice(nodes)
                    break
        return node

    def update(self, states, actions, rewards, next_states, graphs, dones) -> float:
        states = torch.tensor([s for state in states for s in state], dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        next_states = torch.tensor(
            [s for state in next_states for s in state], dtype=torch.float, device=self.device
        ).view(-1)
        dones = torch.tensor(dones, dtype=torch.int, device=self.device)

        data = get_q_net_input(graphs, self.num_features, self.device)

        self.q_net.train()
        actions_q_values = self.q_net(data.x, data.edge_index, data.edge_weight, data.batch, states)
        bidx = torch.unique(data.batch)
        len_b = 0
        for b in bidx:
            actions[b] += len_b
            len_b += (data.batch == b).sum()

        q_values = actions_q_values.gather(dim=0, index=actions)
        self.target_q_net.eval()
        with torch.no_grad():
            if self.ntype == 'DQN':
                max_q_values = torch_scatter.scatter_max(
                    self.target_q_net(data.x, data.edge_index, data.edge_weight, data.batch, next_states)
                    + next_states * -1e6,
                    data.batch
                )[0].clamp_(min=0)
            else:
                max_actions = torch_scatter.scatter_max(actions_q_values + next_states * -1e6, data.batch)[1]
                max_q_values = (
                    self.target_q_net(data.x, data.edge_index, data.edge_weight, data.batch, next_states)
                    .gather(dim=0, index=max_actions)
                ).clamp_(min=0)
            teacher_q_values = self.teacher(data.x, data.edge_index, data.edge_weight, data.batch, states)

        q_targets = rewards + self.gamma ** self.n_steps * max_q_values * (1 - dones)

        self.optim.zero_grad()
        loss1 = F.mse_loss(q_values, q_targets.detach())
        loss2 = kl_divergence_with_scatter(actions_q_values, teacher_q_values.detach(), data.batch)

        loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        loss.backward()
        self.optim.step()

        self.count += 1
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)


def get_q_net_input(graphs: List['IMGraph'], num_features: int, device: torch.device) -> Batch:
    """准备 Q 网络输入数据。"""
    data_list = []
    for graph in graphs:
        edge_index = [[], []]
        edge_weight = []
        for u in range(graph.num_nodes):
            for v, w in graph.out_neighbors_with_weights(u):
                edge_index[0].append(u)
                edge_index[1].append(v)
                edge_weight.append(w)

        x = torch.ones(graph.num_nodes, 2 * num_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        data = Data(x, edge_index, edge_weight=edge_weight)
        data_list.append(data)

    batched_data = Batch.from_data_list(data_list)
    return batched_data.to(device)


def kl_divergence_with_scatter(p, q, batch, eps=1e-15) -> torch.Tensor:
    """计算 KL 散度。"""
    p = p.float()
    q = q.float()

    p_soft = torch_scatter.scatter_softmax(p, dim=0, index=batch)
    q_soft = torch_scatter.scatter_softmax(q, dim=0, index=batch)

    p_safe = p_soft.clamp(min=eps)
    q_safe = q_soft.clamp(min=eps)

    kl_div = (p_safe * (torch.log(p_safe) - torch.log(q_safe))).sum()

    return kl_div
