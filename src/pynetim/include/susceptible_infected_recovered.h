#pragma once
#include "diffusion_model_base.h"

namespace pynetim {

class SusceptibleInfectedRecoveredModel : public DiffusionModelBase {
private:
    double beta;
    double gamma;

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
        std::vector<int> last_infected;
        std::vector<char> in_last(num_nodes, 0);

        int count = 0;

        for (int s : trial_seeds) {
            state[s] = 1;
            infected.push_back(s);
            last_infected.push_back(s);
            in_last[s] = 1;
            count++;
            if (activated_nodes != nullptr) {
                activated_nodes->insert(s);
            }
            if (activation_count != nullptr) {
                (*activation_count)[s]++;
            }
        }

        while (!infected.empty()) {
            std::vector<int> survivors;
            survivors.reserve(infected.size());

            for (int u : last_infected) {
                if (dist(rng) < gamma) {
                    state[u] = 2;
                    in_last[u] = 0;
                } else {
                    survivors.push_back(u);
                }
            }

            for (int u : infected) {
                if (state[u] == 1 && !in_last[u]) {
                    survivors.push_back(u);
                }
            }

            if (survivors.empty()) break;

            std::vector<int> new_infected;
            new_infected.reserve(survivors.size());

            for (int u : survivors) {
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

            last_infected.swap(survivors);
            infected = last_infected;
            infected.insert(infected.end(), new_infected.begin(), new_infected.end());
        }

        return count;
    }

public:
    SusceptibleInfectedRecoveredModel(
        std::shared_ptr<Graph> graph_ptr,
        const std::set<int>& seeds,
        double beta,
        double gamma,
        bool record_activated = false,
        bool record_activation_frequency = false)
        : DiffusionModelBase(graph_ptr, seeds, record_activated, record_activation_frequency),
          beta(beta), gamma(gamma) {

        if (beta <= 0.0 || beta > 1.0) {
            throw std::invalid_argument("beta must be in (0,1]");
        }
        if (gamma <= 0.0 || gamma > 1.0) {
            throw std::invalid_argument("gamma must be in (0,1]");
        }
    }

    void set_beta(double new_beta) {
        if (new_beta <= 0.0 || new_beta > 1.0) {
            throw std::invalid_argument("beta must be in (0,1]");
        }
        beta = new_beta;
    }

    void set_gamma(double new_gamma) {
        if (new_gamma <= 0.0 || new_gamma > 1.0) {
            throw std::invalid_argument("gamma must be in (0,1]");
        }
        gamma = new_gamma;
    }
};

}