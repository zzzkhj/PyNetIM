#pragma once

#include <vector>
#include <queue>
#include <random>
#include <memory>
#include "graph/Graph.h"

namespace pynetim {
namespace utils {

inline std::vector<int> sampleRRSetIC(std::shared_ptr<Graph> graph, std::mt19937& rng) {
    std::vector<int> rr_set;
    std::vector<bool> visited(graph->num_nodes, false);
    std::queue<int> q;

    int start = std::uniform_int_distribution<int>(0, graph->num_nodes - 1)(rng);
    rr_set.push_back(start);
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : graph->in_neighbors(u)) {
            if (visited[v]) continue;
            double w = graph->get_edge_weight(v, u);
            if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) <= w) {
                visited[v] = true;
                rr_set.push_back(v);
                q.push(v);
            }
        }
    }
    return rr_set;
}

inline std::vector<int> sampleRRSetLT(std::shared_ptr<Graph> graph, std::mt19937& rng) {
    std::vector<int> rr_set;
    std::vector<bool> visited(graph->num_nodes, false);

    int current = std::uniform_int_distribution<int>(0, graph->num_nodes - 1)(rng);
    rr_set.push_back(current);
    visited[current] = true;

    while (true) {
        auto in_nbrs = graph->in_neighbors(current);
        if (in_nbrs.empty()) break;

        double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        double cum = 0.0;
        int selected = -1;
        for (int v : in_nbrs) {
            cum += graph->get_edge_weight(v, current);
            if (r <= cum) { selected = v; break; }
        }
        if (selected == -1 || visited[selected]) break;
        visited[selected] = true;
        rr_set.push_back(selected);
        current = selected;
    }
    return rr_set;
}

inline std::vector<std::vector<int>> generateRRSets(
    std::shared_ptr<Graph> graph,
    int num_sets,
    const std::string& model,
    std::optional<int> seed = std::nullopt
) {
    std::mt19937 rng;
    if (seed.has_value()) {
        rng.seed(seed.value());
    } else {
        std::random_device rd;
        rng.seed(rd());
    }

    std::vector<std::vector<int>> rr_sets;
    rr_sets.reserve(num_sets);

    for (int i = 0; i < num_sets; i++) {
        if (model == "IC") {
            rr_sets.push_back(sampleRRSetIC(graph, rng));
        } else {
            rr_sets.push_back(sampleRRSetLT(graph, rng));
        }
    }
    return rr_sets;
}

}
}
