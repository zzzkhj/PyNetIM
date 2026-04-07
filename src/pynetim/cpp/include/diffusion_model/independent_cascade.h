#pragma once
#include "diffusion_model_base.h"

namespace pynetim {

class IndependentCascadeModel : public DiffusionModelBase {
private:
    py::object py_graph_ref;

protected:
    int run_single_trial(
        std::mt19937& rng,
        std::uniform_real_distribution<double>& dist,
        const std::set<int>& trial_seeds,
        std::set<int>* activated_nodes = nullptr,
        std::vector<int>* activation_count = nullptr) const override {

        std::vector<char> activated(num_nodes, 0);
        std::vector<int> q;
        q.reserve(num_nodes);

        int count = 0;

        for (int s : trial_seeds) {
            activated[s] = 1;
            q.push_back(s);
            count++;
            if (activated_nodes != nullptr) {
                activated_nodes->insert(s);
            }
            if (activation_count != nullptr) {
                (*activation_count)[s]++;
            }
        }

        size_t front = 0;

        while (front < q.size()) {
            int u = q[front++];
            const auto& neighbors = graph->out_neighbors(u);

            for (const auto& edge : neighbors) {
                int v = edge.to;
                double w = edge.weight;

                if (activated[v] == 0 && dist(rng) < w) {
                    activated[v] = 1;
                    q.push_back(v);
                    count++;
                    if (activated_nodes != nullptr) {
                        activated_nodes->insert(v);
                    }
                    if (activation_count != nullptr) {
                        (*activation_count)[v]++;
                    }
                }
            }
        }

        return count;
    }

public:
    IndependentCascadeModel(
        std::shared_ptr<Graph> graph_ptr,
        const std::set<int>& seeds,
        py::object py_graph = py::none(),
        bool record_activated = false,
        bool record_activation_frequency = false)
        : DiffusionModelBase(graph_ptr, seeds, record_activated, record_activation_frequency),
          py_graph_ref(py_graph) {}
};

}
