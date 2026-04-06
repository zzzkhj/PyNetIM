#pragma once
#include "diffusion_model_base.h"

namespace pynetim {

class LinearThresholdModel : public DiffusionModelBase {
private:
    double theta_l;
    double theta_h;

protected:
    int run_single_trial(
        std::mt19937& rng,
        std::uniform_real_distribution<double>& dist,
        const std::set<int>& trial_seeds,
        std::set<int>* activated_nodes = nullptr,
        std::vector<int>* activation_count = nullptr) const override {

        std::vector<double> threshold(num_nodes);
        std::vector<char> activated(num_nodes, 0);
        std::vector<double> influence(num_nodes, 0.0);
        std::vector<int> q;

        q.reserve(num_nodes);

        for (int i = 0; i < num_nodes; ++i) {
            threshold[i] = theta_l + dist(rng) * (theta_h - theta_l);
        }

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

                if (activated[v]) continue;

                influence[v] += w;

                if (influence[v] >= threshold[v]) {
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
    LinearThresholdModel(
        std::shared_ptr<Graph> graph_ptr,
        const std::set<int>& seeds,
        double theta_l = 0.0,
        double theta_h = 1.0,
        bool record_activated = false,
        bool record_activation_frequency = false)
        : DiffusionModelBase(graph_ptr, seeds, record_activated, record_activation_frequency),
          theta_l(theta_l), theta_h(theta_h) {

        if (theta_l < 0.0 || theta_l > 1.0) {
            throw std::invalid_argument("theta_l must be in [0,1]");
        }
        if (theta_h < 0.0 || theta_h > 1.0) {
            throw std::invalid_argument("theta_h must be in [0,1]");
        }
        if (theta_l > theta_h) {
            throw std::invalid_argument("theta_l cannot be greater than theta_h");
        }
    }
};

}