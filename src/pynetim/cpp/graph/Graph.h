#pragma once
#include <vector>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include <format>

/* ================== pair<int,int> hash ================== */

struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const noexcept {
        return (static_cast<size_t>(p.first) << 32)
            ^ static_cast<size_t>(p.second);
    }
};

/* ================== Graph ================== */

class Graph {
public:
    int num_nodes;
    int num_edges;
    bool directed;

    // 出邻接表: u -> {v1, v2, ...}
    std::vector<std::unordered_set<int>> adj;

    // 入邻接表（仅有向图）
    std::vector<std::unordered_set<int>> rev_adj;

    // 边权表: (u, v) -> w
    std::unordered_map<std::pair<int, int>, double, PairHash> edges;

public:
    /* ================== 构造函数 ================== */

    Graph(int n,
        const std::vector<std::tuple<int, int>>& edges,
        const std::vector<double>& weights = {},
        bool directed = true)
        : num_nodes(n), num_edges(0), directed(directed) {

        adj.resize(n);
        if (directed) {
            rev_adj.resize(n);
        }

        add_edges(edges, weights);
    }

    /* ================== 构建图 ================== */

    void add_edge(int u, int v, double w = 1.0) {
        // 若已存在边，仅更新权重，不重复计数
        if (edges.find({ u, v }) != edges.end()) {
            edges[{u, v}] = w;
            return;
        }

        adj[u].emplace(v);
        edges.emplace(std::make_pair(u, v), w);
        num_edges++;

        if (directed) {
            rev_adj[v].emplace(u);
        }
        else {
            adj[v].emplace(u);
            edges.emplace(std::make_pair(v, u), w);
        }
    }

    void add_edges(const std::vector<std::tuple<int, int>>& edges,
        const std::vector<double>& weights = {}) {
        if (!weights.empty() && edges.size() != weights.size()) {
            throw std::runtime_error("edges 和 weights 长度不一致");
        }

        for (size_t i = 0; i < edges.size(); ++i) {
            int u = std::get<0>(edges[i]);
            int v = std::get<1>(edges[i]);
            double w = weights.empty() ? 1.0 : weights[i];
            add_edge(u, v, w);
        }
    }

    void update_edge_weight(int u, int v, double w) {
        if (edges.find({ u, v }) == edges.end()) {
            throw std::runtime_error(
                std::format("Edge ({}, {}) does not exist", u, v));
        }
        edges[{u, v}] = w;
    }

    /* ================== 删除边 ================== */

    void remove_edge(int u, int v) {
        auto it = edges.find({ u, v });
        if (it == edges.end()) {
            throw std::runtime_error(
                std::format("Edge ({}, {}) does not exist", u, v));
        }

        adj[u].erase(v);

        if (directed) {
            rev_adj[v].erase(u);
        }
        else {
            adj[v].erase(u);
            edges.erase({ v, u });
        }

        edges.erase(it);
        num_edges--;
    }

    void remove_edges(const std::vector<std::tuple<int, int>>& edges_) {
        for (const auto& e : edges_) {
            int u = std::get<0>(e);
            int v = std::get<1>(e);
            remove_edge(u, v);
        }
    }

    /* ================== 查询接口 ================== */

    const std::unordered_set<int>& out_neighbors(int u) const {
        return adj[u];
    }

    const std::unordered_set<int>& in_neighbors(int u) const {
        return directed ? rev_adj[u] : adj[u];
    }

    int out_degree(int u) const {
        return static_cast<int>(adj[u].size());
    }

    int in_degree(int u) const {
        return static_cast<int>(directed ? rev_adj[u].size() : adj[u].size());
    }

    int degree(int u) const {
        return out_degree(u);
    }

    const std::vector<std::unordered_set<int>>& get_adj_list() const {
        return adj;
    }

    std::vector<std::vector<double>> get_adj_matrix() const {
        std::vector<std::vector<double>> adj_matrix(
            num_nodes, std::vector<double>(num_nodes, 0.0));

        for (int u = 0; u < num_nodes; ++u) {
            for (int v : adj[u]) {
                adj_matrix[u][v] = edges.at({ u, v });
            }
        }
        return adj_matrix;
    }

    std::string __repr__() const {
        return std::format("{} graph with {} nodes and {} edges",
            directed ? "Directed" : "Undirected",
            num_nodes, num_edges);
    }
};
