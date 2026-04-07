#pragma once

#include <vector>
#include <queue>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>
#include <iostream>
#include <set>
#include <optional>
#include "../graph/Graph.h"

namespace pynetim {

class BaseRISAlgorithm {
protected:
    std::shared_ptr<Graph> graph_;
    std::string model_;
    std::mt19937 rng_;
    bool verbose_;

    std::vector<std::vector<int>> hyperGT_;
    std::vector<std::vector<int>> hyperG_;
    std::set<int> seedSet_;

    std::vector<int> sampleRRSetIC() {
        std::vector<int> rr_set;
        std::vector<bool> visited(graph_->num_nodes, false);
        std::queue<int> q;

        int start = std::uniform_int_distribution<int>(
            0, graph_->num_nodes - 1)(rng_);
        rr_set.push_back(start);
        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : graph_->in_neighbors(u)) {
                if (visited[v]) continue;
                double w = graph_->get_edge_weight(v, u);
                if (std::uniform_real_distribution<double>(0.0, 1.0)(rng_) <= w) {
                    visited[v] = true;
                    rr_set.push_back(v);
                    q.push(v);
                }
            }
        }
        return rr_set;
    }

    std::vector<int> sampleRRSetLT() {
        std::vector<int> rr_set;
        std::vector<bool> visited(graph_->num_nodes, false);

        int current = std::uniform_int_distribution<int>(
            0, graph_->num_nodes - 1)(rng_);
        rr_set.push_back(current);
        visited[current] = true;

        while (true) {
            auto in_nbrs = graph_->in_neighbors(current);
            if (in_nbrs.empty()) break;

            double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
            double cum = 0.0;
            int selected = -1;
            for (int v : in_nbrs) {
                cum += graph_->get_edge_weight(v, current);
                if (r <= cum) { selected = v; break; }
            }
            if (selected == -1 || visited[selected]) break;
            visited[selected] = true;
            rr_set.push_back(selected);
            current = selected;
        }
        return rr_set;
    }

    void buildHyperGraphR(int64_t R) {
        int prev_size = static_cast<int>(hyperGT_.size());
        if (R <= prev_size) return;

        hyperGT_.resize(R);
        hyperG_.resize(graph_->num_nodes);

        for (int i = prev_size; i < static_cast<int>(R); i++) {
            hyperGT_[i] = (model_ == "IC") ? sampleRRSetIC() : sampleRRSetLT();
            for (int node : hyperGT_[i]) {
                hyperG_[node].push_back(i);
            }
        }
    }

    double buildSeedSet(int k) {
        int64_t num_rr = static_cast<int64_t>(hyperGT_.size());

        std::vector<int> coverage(graph_->num_nodes, 0);
        for (int i = 0; i < graph_->num_nodes; i++) {
            coverage[i] = static_cast<int>(hyperG_[i].size());
        }

        std::priority_queue<std::pair<int,int>> heap;
        for (int i = 0; i < graph_->num_nodes; i++) {
            heap.push({coverage[i], i});
        }

        std::vector<bool> edge_mark(num_rr, false);
        std::vector<bool> node_mark(graph_->num_nodes, true);
        seedSet_.clear();
        int64_t influence = 0;

        while (static_cast<int>(seedSet_.size()) < k) {
            auto [cov, node] = heap.top(); heap.pop();
            if (cov > coverage[node]) {
                heap.push({coverage[node], node});
                continue;
            }
            influence += coverage[node];
            seedSet_.insert(node);
            node_mark[node] = false;

            for (int rr_idx : hyperG_[node]) {
                if (edge_mark[rr_idx]) continue;
                edge_mark[rr_idx] = true;
                for (int n : hyperGT_[rr_idx]) {
                    if (node_mark[n]) coverage[n]--;
                }
            }
        }
        return static_cast<double>(influence) / static_cast<double>(num_rr);
    }

    static double logCnk(int n, int k) {
        double ans = 0.0;
        for (int i = n - k + 1; i <= n; i++) ans += std::log(static_cast<double>(i));
        for (int i = 1; i <= k; i++)          ans -= std::log(static_cast<double>(i));
        return ans;
    }

public:
    BaseRISAlgorithm(std::shared_ptr<Graph> graph,
                     const std::string& model,
                     std::optional<int> seed = std::nullopt,
                     bool verbose = false)
        : graph_(graph), model_(model), verbose_(verbose) {
        if (model_ != "IC" && model_ != "LT")
            throw std::invalid_argument("模型必须是 'IC' 或 'LT'");
        
        if (seed.has_value()) {
            rng_.seed(seed.value());
        } else {
            std::random_device rd;
            rng_.seed(rd());
        }
    }

    virtual ~BaseRISAlgorithm() = default;

    virtual std::set<int> run(int k, int num_rr_sets) {
        if (k <= 0 || num_rr_sets <= 0) return {};
        hyperGT_.clear();
        hyperG_.clear();
        hyperG_.resize(graph_->num_nodes);
        seedSet_.clear();
        buildHyperGraphR(num_rr_sets);
        buildSeedSet(k);
        return seedSet_;
    }

    std::set<int> getSeeds() const { return seedSet_; }
};

} // namespace pynetim
