#pragma once
#include "diffusion_model_base.h"

namespace pynetim {

class SusceptibleInfectedModel : public DiffusionModelBase {
private:
    double beta;

protected:
    int run_single_trial(
        std::mt19937& rng,
        std::uniform_real_distribution<double>& dist,
        const std::set<int>& trial_seeds,
        std::set<int>* activated_nodes = nullptr,
        std::vector<int>* activation_count = nullptr) const override {

        std::vector<char> state(num_nodes, 0);
        std::vector<int> infected;
        infected.reserve(num_nodes);

        int count = 0;

        for (int s : trial_seeds) {
            state[s] = 1;
            infected.push_back(s);
            count++;
            if (activated_nodes != nullptr) {
                activated_nodes->insert(s);
            }
            if (activation_count != nullptr) {
                (*activation_count)[s]++;
            }
        }

        while (true) {
            std::vector<int> new_infected;
            new_infected.reserve(infected.size());

            for (int u : infected) {
                const auto& neighbors = graph->out_neighbors(u);
                for (const auto& edge : neighbors) {
                    int v = edge.to;
                    if (state[v] == 0 && dist(rng) < beta) {
                        state[v] = 1;
                        new_infected.push_back(v);
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

            if (new_infected.empty()) break;

            infected = std::move(new_infected);
        }

        return count;
    }

public:
    SusceptibleInfectedModel(
        std::shared_ptr<Graph> graph_ptr,
        const std::set<int>& seeds,
        double beta,
        int max_steps = 0,
        bool record_activated = false,
        bool record_activation_frequency = false)
        : DiffusionModelBase(graph_ptr, seeds, record_activated, record_activation_frequency),
          beta(beta) {
        (void)max_steps;

        if (beta <= 0.0 || beta > 1.0) {
            throw std::invalid_argument("beta 必须在 (0,1] 范围内");
        }
    }

    void set_beta(double new_beta) {
        if (new_beta <= 0.0 || new_beta > 1.0) {
            throw std::invalid_argument("beta 必须在 (0,1] 范围内");
        }
        beta = new_beta;
    }

    void set_max_steps(int steps) {
        (void)steps;
    }
};

}
