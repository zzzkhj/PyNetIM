#pragma once
#include <vector>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include <format>
#include <memory>
#include <algorithm>

namespace pynetim {

struct EdgeHash {
    size_t operator()(const std::pair<int, int>& p) const noexcept {
        return (static_cast<size_t>(p.first) << 32)
            ^ static_cast<size_t>(p.second);
    }
};

struct Edge {
    int to;
    double weight;

    Edge(int t, double w) : to(t), weight(w) {}
};

class Graph {
public:
    int num_nodes;
    int num_edges;
    bool directed;
    std::unordered_map<std::pair<int, int>, double, EdgeHash> edges;

private:
    std::vector<std::vector<Edge>> adj;
    std::vector<std::vector<int>> rev_adj;

    std::vector<int> out_degree_cache;
    std::vector<int> in_degree_cache;
    bool degrees_dirty;

    void update_degree_cache() {
        if (!degrees_dirty) return;
        
        out_degree_cache.resize(num_nodes);
        in_degree_cache.resize(num_nodes);
        
        for (int u = 0; u < num_nodes; ++u) {
            out_degree_cache[u] = static_cast<int>(adj[u].size());
            in_degree_cache[u] = static_cast<int>(rev_adj[u].size());
        }
        
        degrees_dirty = false;
    }

public:
    Graph(int n,
        const std::vector<std::tuple<int, int>>& edges_list,
        const std::vector<double>& weights = {},
        bool directed = true)
        : num_nodes(n), num_edges(0), directed(directed), degrees_dirty(true) {

        adj.resize(n);
        rev_adj.resize(n);
        out_degree_cache.resize(n, 0);
        in_degree_cache.resize(n, 0);

        add_edges(edges_list, weights);
    }

    Graph(const Graph& other) = delete;
    Graph& operator=(const Graph& other) = delete;

    Graph(Graph&& other) noexcept
        : num_nodes(other.num_nodes),
          num_edges(other.num_edges),
          directed(other.directed),
          edges(std::move(other.edges)),
          adj(std::move(other.adj)),
          rev_adj(std::move(other.rev_adj)),
          out_degree_cache(std::move(other.out_degree_cache)),
          in_degree_cache(std::move(other.in_degree_cache)),
          degrees_dirty(other.degrees_dirty) {
        other.num_nodes = 0;
        other.num_edges = 0;
    }

    Graph& operator=(Graph&& other) noexcept {
        if (this != &other) {
            num_nodes = other.num_nodes;
            num_edges = other.num_edges;
            directed = other.directed;
            adj = std::move(other.adj);
            rev_adj = std::move(other.rev_adj);
            edges = std::move(other.edges);
            out_degree_cache = std::move(other.out_degree_cache);
            in_degree_cache = std::move(other.in_degree_cache);
            degrees_dirty = other.degrees_dirty;

            other.num_nodes = 0;
            other.num_edges = 0;
        }
        return *this;
    }

    void add_edge(int u, int v, double w = 1.0) {
        if (edges.find({ u, v }) != edges.end()) {
            edges[{u, v}] = w;
            for (auto& edge : adj[u]) {
                if (edge.to == v) {
                    edge.weight = w;
                    break;
                }
            }
            return;
        }

        adj[u].emplace_back(v, w);
        edges.emplace(std::make_pair(u, v), w);
        num_edges++;

        if (directed) {
            rev_adj[v].push_back(u);
        }
        else {
            adj[v].emplace_back(u, w);
            edges.emplace(std::make_pair(v, u), w);
        }

        degrees_dirty = true;
    }

    void add_edges(const std::vector<std::tuple<int, int>>& edges_list,
        const std::vector<double>& weights = {}) {
        if (!weights.empty() && edges_list.size() != weights.size()) {
            throw std::runtime_error("edges and weights length mismatch");
        }

        for (size_t i = 0; i < edges_list.size(); ++i) {
            int u = std::get<0>(edges_list[i]);
            int v = std::get<1>(edges_list[i]);
            double w = weights.empty() ? 1.0 : weights[i];
            add_edge(u, v, w);
        }
    }

    void update_edge_weight(int u, int v, double w) {
        auto it = edges.find({ u, v });
        if (it == edges.end()) {
            throw std::runtime_error(
                std::format("Edge ({}, {}) does not exist", u, v));
        }
        edges[{u, v}] = w;
        for (auto& edge : adj[u]) {
            if (edge.to == v) {
                edge.weight = w;
                break;
            }
        }
    }

    void remove_edge(int u, int v) {
        auto it = edges.find({ u, v });
        if (it == edges.end()) {
            throw std::runtime_error(
                std::format("Edge ({}, {}) does not exist", u, v));
        }

        auto& neighbors = adj[u];
        neighbors.erase(
            std::remove_if(neighbors.begin(), neighbors.end(),
                [v](const Edge& e) { return e.to == v; }),
            neighbors.end());

        if (directed) {
            auto& rev_neighbors = rev_adj[v];
            rev_neighbors.erase(
                std::remove(rev_neighbors.begin(), rev_neighbors.end(), u),
                rev_neighbors.end());
        }
        else {
            auto& rev_neighbors = adj[v];
            rev_neighbors.erase(
                std::remove_if(rev_neighbors.begin(), rev_neighbors.end(),
                    [u](const Edge& e) { return e.to == u; }),
                rev_neighbors.end());
            edges.erase({ v, u });
        }

        edges.erase(it);
        num_edges--;
        degrees_dirty = true;
    }

    void remove_edges(const std::vector<std::tuple<int, int>>& edges_list) {
        for (const auto& e : edges_list) {
            int u = std::get<0>(e);
            int v = std::get<1>(e);
            remove_edge(u, v);
        }
    }

    const std::vector<Edge>& out_neighbors(int u) const {
        return adj[u];
    }

    const std::vector<int>& in_neighbors(int u) const {
        return directed ? rev_adj[u] : reinterpret_cast<const std::vector<int>&>(adj[u]);
    }

    int out_degree(int u) const {
        return static_cast<int>(adj[u].size());
    }

    int in_degree(int u) const {
        return directed ? static_cast<int>(rev_adj[u].size()) : static_cast<int>(adj[u].size());
    }

    int degree(int u) const {
        return directed ? in_degree(u) + out_degree(u) : out_degree(u);
    }

    std::vector<int> get_all_degrees() const {
        std::vector<int> degree_vector(num_nodes, 0);
        for (int u = 0; u < num_nodes; u++) {
            degree_vector[u] = degree(u);
        }
        return degree_vector;
    }

    std::vector<int> get_all_in_degrees() const {
        std::vector<int> degree_vector(num_nodes, 0);
        for (int u = 0; u < num_nodes; u++) {
            degree_vector[u] = in_degree(u);
        }
        return degree_vector;
    }

    std::vector<int> get_all_out_degrees() const {
        std::vector<int> degree_vector(num_nodes, 0);
        for (int u = 0; u < num_nodes; u++) {
            degree_vector[u] = out_degree(u);
        }
        return degree_vector;
    }

    std::vector<int> batch_out_degree(const std::vector<int>& nodes) const {
        std::vector<int> degrees;
        degrees.reserve(nodes.size());
        for (int u : nodes) {
            degrees.push_back(out_degree(u));
        }
        return degrees;
    }

    std::vector<int> batch_in_degree(const std::vector<int>& nodes) const {
        std::vector<int> degrees;
        degrees.reserve(nodes.size());
        for (int u : nodes) {
            degrees.push_back(in_degree(u));
        }
        return degrees;
    }

    std::vector<int> batch_degree(const std::vector<int>& nodes) const {
        std::vector<int> degrees;
        degrees.reserve(nodes.size());
        for (int u : nodes) {
            degrees.push_back(degree(u));
        }
        return degrees;
    }

    std::vector<std::vector<std::tuple<int, double>>> batch_out_neighbors(const std::vector<int>& nodes) const {
        std::vector<std::vector<std::tuple<int, double>>> result;
        result.reserve(nodes.size());
        for (int u : nodes) {
            std::vector<std::tuple<int, double>> neighbors;
            neighbors.reserve(adj[u].size());
            for (const auto& edge : adj[u]) {
                neighbors.emplace_back(edge.to, edge.weight);
            }
            result.push_back(std::move(neighbors));
        }
        return result;
    }

    std::vector<double> batch_get_edge_weight(const std::vector<std::tuple<int, int>>& edge_list) const {
        std::vector<double> weights;
        weights.reserve(edge_list.size());
        for (const auto& e : edge_list) {
            int u = std::get<0>(e);
            int v = std::get<1>(e);
            weights.push_back(get_edge_weight(u, v));
        }
        return weights;
    }

    const std::vector<std::vector<Edge>>& get_adj_list() const {
        return adj;
    }

    std::vector<std::vector<double>> get_adj_matrix() const {
        std::vector<std::vector<double>> adj_matrix(
            num_nodes, std::vector<double>(num_nodes, 0.0));

        for (int u = 0; u < num_nodes; ++u) {
            for (const auto& edge : adj[u]) {
                adj_matrix[u][edge.to] = edge.weight;
            }
        }
        return adj_matrix;
    }

    std::vector<std::tuple<int, int, double>> get_adj_matrix_sparse() const {
        std::vector<std::tuple<int, int, double>> sparse_matrix;
        sparse_matrix.reserve(num_edges);

        for (int u = 0; u < num_nodes; ++u) {
            for (const auto& edge : adj[u]) {
                sparse_matrix.emplace_back(u, edge.to, edge.weight);
            }
        }
        return sparse_matrix;
    }

    double get_edge_weight(int u, int v) const {
        auto it = edges.find({ u, v });
        if (it == edges.end()) {
            return 0.0;
        }
        return it->second;
    }

    std::string __repr__() const {
        return std::format("{} graph with {} nodes and {} edges",
            directed ? "Directed" : "Undirected",
            num_nodes, num_edges);
    }
};

}