import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_softmax, scatter_max
from torch.nn import Embedding
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynetim import IMGraph

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)

EPS = 1e-15


class S2V_DQN(nn.Module):
    """Structure2Vec DQN 神经网络模型。

    References:
        Dai H, Khalil E B, Zhang Y, Dilkina B, Song L. Learning Combinatorial Optimization
        Algorithms over Graphs. Advances in Neural Information Processing Systems (NeurIPS), 2017.
    """

    def __init__(self, reg_hidden, embed_dim, node_dim, edge_dim, T, w_scale, avg=False):
        super(S2V_DQN, self).__init__()
        self.T = T
        self.embed_dim = embed_dim
        self.reg_hidden = reg_hidden
        self.avg = avg

        self.w_n2l = torch.nn.Parameter(torch.Tensor(node_dim, embed_dim))
        torch.nn.init.normal_(self.w_n2l, mean=0, std=w_scale)

        self.w_e2l = torch.nn.Parameter(torch.Tensor(edge_dim, embed_dim))
        torch.nn.init.normal_(self.w_e2l, mean=0, std=w_scale)

        self.p_node_conv = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.p_node_conv, mean=0, std=w_scale)

        self.trans_node_1 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.trans_node_1, mean=0, std=w_scale)

        self.trans_node_2 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.trans_node_2, mean=0, std=w_scale)

        if self.reg_hidden > 0:
            self.h1_weight = torch.nn.Parameter(torch.Tensor(2 * embed_dim, reg_hidden))
            torch.nn.init.normal_(self.h1_weight, mean=0, std=w_scale)
            self.h2_weight = torch.nn.Parameter(torch.Tensor(reg_hidden, 1))
            torch.nn.init.normal_(self.h2_weight, mean=0, std=w_scale)
            self.last_w = self.h2_weight
        else:
            self.h1_weight = torch.nn.Parameter(torch.Tensor(2 * embed_dim, 1))
            torch.nn.init.normal_(self.h1_weight, mean=0, std=w_scale)
            self.last_w = self.h1_weight

        self.scatter_aggr = (scatter_mean if self.avg else scatter_add)

    def forward(self, data):
        data.x = torch.matmul(data.x, self.w_n2l)
        data.x = F.relu(data.x)
        data.edge_attr = torch.matmul(data.edge_attr, self.w_e2l)

        for _ in range(self.T):
            msg_linear = torch.matmul(data.x, self.p_node_conv)
            n2e_linear = msg_linear[data.edge_index[0]]
            edge_rep = torch.add(n2e_linear, data.edge_attr)
            edge_rep = F.relu(edge_rep)
            e2n = self.scatter_aggr(edge_rep, data.edge_index[1], dim=0, dim_size=data.x.size(0))
            data.x = torch.add(torch.matmul(e2n, self.trans_node_1),
                               torch.matmul(data.x, self.trans_node_2))
            data.x = F.relu(data.x)

        y_potential = self.scatter_aggr(data.x, data.batch, dim=0)
        if data.y is not None:
            action_embed = data.x[data.y]
            embed_s_a = torch.cat((action_embed, y_potential), dim=-1)
            last_output = embed_s_a
            if self.reg_hidden > 0:
                hidden = torch.matmul(embed_s_a, self.h1_weight)
                last_output = F.relu(hidden)
            q_pred = torch.matmul(last_output, self.last_w)
            return q_pred
        else:
            rep_y = y_potential[data.batch]
            embed_s_a_all = torch.cat((data.x, rep_y), dim=-1)
            last_output = embed_s_a_all
            if self.reg_hidden > 0:
                hidden = torch.matmul(embed_s_a_all, self.h1_weight)
                last_output = torch.relu(hidden)
            q_on_all = torch.matmul(last_output, self.last_w)
            return q_on_all


class Tripling(nn.Module):
    """Tripling 三重门控图神经网络模型。

    References:
        Chen T, Yan S, Guo J, Wu W. ToupleGDD: A Fine-Designed Solution of Influence
        Maximization by Deep Reinforcement Learning. IEEE Transactions on Computational
        Social Systems, 2024.
    """

    def __init__(self, embed_dim, sgate_l1_dim, tgate_l1_dim, T, hidden_dims, w_scale):
        super(Tripling, self).__init__()
        self.embed_dim = embed_dim
        self.sgate_l1_dim = sgate_l1_dim
        self.tgate_l1_dim = tgate_l1_dim
        self.T = T
        self.hidden_dims = hidden_dims.copy()
        self.hidden_dims.insert(0, embed_dim)
        self.trans_weights = nn.ParameterList()

        self.influgate_etas = nn.ParameterList()
        self.state_weights_self = nn.ParameterList()
        self.state_weights_neibor = nn.ParameterList()
        self.state_weights_attention = nn.ParameterList()
        self.state_weights_edge = nn.ParameterList()

        self.source_betas = nn.ParameterList()
        self.sourcegate_layer1s = nn.ModuleList()
        self.sourcegate_layer2s = nn.ModuleList()
        self.source_weights_self = nn.ParameterList()
        self.source_weights_neibor = nn.ParameterList()
        self.source_weights_state = nn.ParameterList()
        self.source_weights_attention = nn.ParameterList()
        self.source_weights_edge = nn.ParameterList()

        self.target_taus = nn.ParameterList()
        self.targetgate_layer1s = nn.ModuleList()
        self.targetgate_layer2s = nn.ModuleList()
        self.target_weights_self = nn.ParameterList()
        self.target_weights_neibor = nn.ParameterList()
        self.target_weights_state = nn.ParameterList()
        self.target_weights_attention = nn.ParameterList()
        self.target_weights_edge = nn.ParameterList()

        for i in range(1, T + 1):
            self.trans_weights.append(torch.nn.Parameter(torch.Tensor(self.hidden_dims[i - 1], self.hidden_dims[i])))
            torch.nn.init.normal_(self.trans_weights[-1], mean=0, std=w_scale)

            self.influgate_etas.append(torch.nn.Parameter(torch.Tensor(2 * self.hidden_dims[i], 1)))
            torch.nn.init.normal_(self.influgate_etas[-1], mean=0, std=w_scale)
            self.state_weights_self.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.state_weights_self[-1], mean=0, std=w_scale)
            self.state_weights_neibor.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.state_weights_neibor[-1], mean=0, std=w_scale)
            self.state_weights_attention.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.state_weights_attention[-1], mean=0, std=w_scale)
            self.state_weights_edge.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.state_weights_edge[-1], mean=0, std=w_scale)

            self.source_betas.append(torch.nn.Parameter(torch.Tensor(2 * self.hidden_dims[i], 1)))
            torch.nn.init.normal_(self.source_betas[-1], mean=0, std=w_scale)
            self.sourcegate_layer1s.append(torch.nn.Linear(self.hidden_dims[i - 1], sgate_l1_dim, True))
            self.sourcegate_layer2s.append(torch.nn.Linear(sgate_l1_dim, self.hidden_dims[i], True))
            self.source_weights_self.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_self[-1], mean=0, std=w_scale)
            self.source_weights_neibor.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_neibor[-1], mean=0, std=w_scale)
            self.source_weights_state.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_state[-1], mean=0, std=w_scale)
            self.source_weights_attention.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_attention[-1], mean=0, std=w_scale)
            self.source_weights_edge.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_edge[-1], mean=0, std=w_scale)

            self.target_taus.append(torch.nn.Parameter(torch.Tensor(2 * self.hidden_dims[i], 1)))
            torch.nn.init.normal_(self.target_taus[-1], mean=0, std=w_scale)
            self.targetgate_layer1s.append(torch.nn.Linear(self.hidden_dims[i - 1], tgate_l1_dim, True))
            self.targetgate_layer2s.append(torch.nn.Linear(tgate_l1_dim, self.hidden_dims[i], True))
            self.target_weights_self.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_self[-1], mean=0, std=w_scale)
            self.target_weights_neibor.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_neibor[-1], mean=0, std=w_scale)
            self.target_weights_state.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_state[-1], mean=0, std=w_scale)
            self.target_weights_attention.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_attention[-1], mean=0, std=w_scale)
            self.target_weights_edge.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_edge[-1], mean=0, std=w_scale)

        self.theta1 = torch.nn.Parameter(torch.Tensor(3 * self.hidden_dims[-1], 1))
        torch.nn.init.normal_(self.theta1, mean=0, std=w_scale)
        self.theta2 = torch.nn.Parameter(torch.Tensor(self.hidden_dims[-1], self.hidden_dims[-1]))
        torch.nn.init.normal_(self.theta2, mean=0, std=w_scale)
        self.theta3 = torch.nn.Parameter(torch.Tensor(self.hidden_dims[-1], self.hidden_dims[-1]))
        torch.nn.init.normal_(self.theta3, mean=0, std=w_scale)
        self.theta4 = torch.nn.Parameter(torch.Tensor(self.hidden_dims[-1], self.hidden_dims[-1]))
        torch.nn.init.normal_(self.theta4, mean=0, std=w_scale)

    def forward(self, data):
        source_influ = data.x[:, :self.hidden_dims[0]]
        target_influ = data.x[:, self.hidden_dims[0]:2 * self.hidden_dims[0]]
        state = data.x[:, -1]
        for i in range(0, self.T):
            trans_source_influ = torch.matmul(source_influ, self.trans_weights[i])
            trans_target_influ = torch.matmul(target_influ, self.trans_weights[i])
            trans_influ = torch.cat((trans_source_influ[data.edge_index[0]], trans_target_influ[data.edge_index[1]]),
                                    dim=-1)

            e_uv = torch.matmul(trans_influ, self.influgate_etas[i]).squeeze(dim=1)
            e_uv = F.leaky_relu(e_uv, 0.2)
            influgate = scatter_softmax(e_uv, data.edge_index[1])
            influgate = influgate * self.state_weights_attention[i] + data.edge_weight * self.state_weights_edge[i]

            a_v = scatter_add(influgate * state[data.edge_index[0]], data.edge_index[1], dim_size=data.x.size(0))
            new_state = torch.sigmoid(state * self.state_weights_self[i] + a_v * self.state_weights_neibor[i])
            new_state = new_state * (1 - data.x[:, -1]) + data.x[:, -1]

            f_vw = torch.matmul(trans_influ, self.source_betas[i]).squeeze(dim=1)
            f_vw = F.leaky_relu(f_vw, 0.2)
            alpha_vw = scatter_softmax(f_vw, data.edge_index[0])
            alpha_vw = alpha_vw * self.source_weights_attention[i] + data.edge_weight * self.source_weights_edge[i]

            sourcegate = F.leaky_relu(
                self.sourcegate_layer2s[i](F.leaky_relu(self.sourcegate_layer1s[i](target_influ[data.edge_index[1]]), 0.2)),
                0.2)
            b_v = scatter_add(alpha_vw.unsqueeze(dim=1) * sourcegate, data.edge_index[0], dim=0, dim_size=data.x.size(0))
            new_source_influ = F.leaky_relu(trans_source_influ * self.source_weights_self[i] +
                                            b_v * self.source_weights_neibor[i] +
                                            (state * self.source_weights_state[i]).unsqueeze(dim=1))

            d_uv = torch.matmul(trans_influ, self.target_taus[i]).squeeze(dim=1)
            d_uv = F.leaky_relu(d_uv, 0.2)
            phi_uv = scatter_softmax(d_uv, data.edge_index[1])
            phi_uv = phi_uv * self.target_weights_attention[i] + data.edge_weight * self.target_weights_edge[i]

            targetgate = F.leaky_relu(
                self.targetgate_layer2s[i](F.leaky_relu(self.targetgate_layer1s[i](source_influ[data.edge_index[0]]), 0.2)),
                0.2)
            c_v = scatter_add(phi_uv.unsqueeze(dim=1) * targetgate, data.edge_index[1], dim=0, dim_size=data.x.size(0))
            new_target_influ = F.leaky_relu(trans_target_influ * self.target_weights_self[i] +
                                            c_v * self.target_weights_neibor[i] +
                                            (state * self.target_weights_state[i]).unsqueeze(dim=1))

            state = new_state
            source_influ = new_source_influ
            target_influ = new_target_influ

        if data.y is not None:
            S_v = source_influ[data.y]
            not_y = torch.ones(target_influ.size(0), dtype=torch.bool, device=data.y.device)
            not_y[data.y] = False
            not_selected = data.x[:, -1] == 0
            not_idx = torch.logical_and(not_y, not_selected)

            batch_idx = data.batch[not_idx]
            T_u = target_influ[not_idx]

            is_idx = data.x[:, -1] == 1
            batch_is_idx = data.batch[is_idx]
            S_w = source_influ[is_idx]

            q_pred = torch.matmul(F.leaky_relu(torch.cat([
                torch.matmul(S_v, self.theta2),
                torch.matmul(scatter_add(S_w, batch_is_idx, dim=0, dim_size=data.batch[-1].item() + 1), self.theta4),
                torch.matmul(scatter_add(T_u, batch_idx, dim=0, dim_size=data.batch[-1].item() + 1), self.theta3)
            ], dim=-1)), self.theta1)

            return q_pred
        else:
            target_influ[data.x[:, -1] == 1] = 0.0
            state[data.x[:, -1] == 1] = 1.0

            source_influ_copy = source_influ.clone()
            source_influ[data.x[:, -1] == 0] = 0.0
            source_influ_w = scatter_add(source_influ, data.batch, dim=0).repeat_interleave(
                scatter_add(torch.ones(data.batch.size(0), dtype=torch.long, device=data.batch.device), data.batch), dim=0)
            source_influ_w[data.x[:, -1] == 1] = 0.0
            source_influ_copy[data.x[:, -1] == 1] = 0.0

            q_on_all = torch.matmul(F.leaky_relu(torch.cat([
                torch.matmul(source_influ_copy, self.theta2),
                torch.matmul(source_influ_w, self.theta4),
                torch.matmul(scatter_add(target_influ, data.batch, dim=0).repeat_interleave(
                    scatter_add(torch.ones(data.batch.size(0), dtype=torch.long, device=data.batch.device), data.batch),
                    dim=0) - target_influ, self.theta3)
            ], dim=-1)), self.theta1)

            return q_on_all


class DeepWalkNeg(nn.Module):
    """DeepWalk 负采样节点嵌入模型。

    References:
        Perozzi B, Al-Rfou R, Skiena S. DeepWalk: Online Learning of Social Representations.
        ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2014.
    """

    def __init__(self, graph: "IMGraph", embedding_dim, walk_length, r_hop, r_hop_size,
                 walks_per_node=1, num_negative_samples=1, restart=0.5, sparse=False):
        super().__init__()

        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.walks_per_node = walks_per_node
        self.r_hop = r_hop
        self.r_hop_size = r_hop_size
        self.restart = restart
        self.num_negative_samples = num_negative_samples

        self.embedding = Embedding(graph.num_nodes, embedding_dim * 2 + 1, sparse=sparse)
        self._children_cache = {}
        self._parents_cache = {}

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, batch=None):
        emb = self.embedding.weight
        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs):
        return DataLoader(range(self.graph.num_nodes),
                          collate_fn=self.sample, **kwargs)

    def _get_children(self, node):
        if node not in self._children_cache:
            self._children_cache[node] = self.graph.out_neighbors(node)
        return self._children_cache[node]

    def _get_parents(self, node):
        if node not in self._parents_cache:
            self._parents_cache[node] = self.graph.in_neighbors(node)
        return self._parents_cache[node]

    def random_walk(self, start, walk_len, restart=0.5, rand=None):
        if rand is None:
            rand = random.Random()
        path = [start]
        for _ in range(walk_len):
            cur = path[-1]
            children = self._get_children(cur)
            if len(children) > 0 and rand.random() >= restart:
                path.append(rand.choice(children))
            else:
                path.append(path[0])
        return path

    def r_hop_neibors(self, start, r_hop):
        neibors = {start}
        frontier = {start}
        for _ in range(r_hop):
            new_frontier = set()
            for node in frontier:
                new_frontier.update(self._get_children(node))
                new_frontier.update(self._get_parents(node))
            frontier = new_frontier - neibors
            neibors.update(frontier)
        return list(neibors)

    def sample(self, batch):
        rand = random.Random()
        pos_rw = []
        neg_rw = []
        for node in batch:
            for _ in range(self.walks_per_node):
                walk = self.random_walk(node, self.walk_length, self.restart, rand)
                r_hop_neibors = self.r_hop_neibors(node, self.r_hop)
                if len(r_hop_neibors) > self.r_hop_size:
                    r_hop_neibors = rand.sample(r_hop_neibors, self.r_hop_size)
                for i in range(len(walk)):
                    for j in range(i + 1, min(i + self.r_hop_size + 1, len(walk) + 1)):
                        pos_rw.append([walk[i], walk[j - 1]])
                        for _ in range(self.num_negative_samples):
                            while True:
                                neg_node = rand.randint(0, self.graph.num_nodes - 1)
                                if neg_node not in r_hop_neibors:
                                    break
                            neg_rw.append([walk[i], neg_node])
        return torch.tensor(pos_rw, dtype=torch.long), torch.tensor(neg_rw, dtype=torch.long)

    def loss(self, pos_rw, neg_rw):
        pos_loss = -torch.log(
            torch.sigmoid((self.embedding(pos_rw[:, 0]) * self.embedding(pos_rw[:, 1])).sum(dim=-1))).mean()
        neg_loss = -torch.log(
            torch.sigmoid(-(self.embedding(neg_rw[:, 0]) * self.embedding(neg_rw[:, 1])).sum(dim=-1))).mean()
        return pos_loss + neg_loss


def get_init_node_embed(graph: "IMGraph", num_epochs, device):
    model = DeepWalkNeg(graph, embedding_dim=50, walk_length=3, r_hop=5, r_hop_size=5,
                        walks_per_node=50, num_negative_samples=5, restart=0.15, sparse=True).to(device)

    loader = model.loader(batch_size=32, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(num_epochs):
        train()

    return model().detach().cpu().clone()
