#pragma once
#include "common.h"

namespace pynetim {

class LinearThresholdModel {
private:
    std::set<int> seeds;
    std::shared_ptr<Graph> graph;
    int num_nodes;
    double theta_l;
    double theta_h;
    bool record_activated;
    mutable std::set<int> last_activated_nodes;
    bool record_activation_frequency;
    mutable std::vector<int> activation_frequency;

    int run_single_trial(std::mt19937& rng,
        std::uniform_real_distribution<double>& dist,
        const std::set<int>& trial_seeds,
        std::set<int>* activated_nodes = nullptr) const {

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
            if (record_activated && activated_nodes != nullptr) {
                activated_nodes->insert(s);
            }
            if (record_activation_frequency) {
                activation_frequency[s]++;
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
                    if (record_activated && activated_nodes != nullptr) {
                        activated_nodes->insert(v);
                    }
                    if (record_activation_frequency) {
                        activation_frequency[v]++;
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
        : seeds(seeds), graph(graph_ptr), num_nodes(graph_ptr->num_nodes),
          theta_l(theta_l), theta_h(theta_h), record_activated(record_activated), 
          record_activation_frequency(record_activation_frequency) {

        activation_frequency.resize(num_nodes, 0);

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

    void set_seeds(const std::set<int>& new_seeds) {
        seeds = new_seeds;
    }

    void set_record_activated(bool record) {
        record_activated = record;
        if (!record) {
            last_activated_nodes.clear();
        }
    }

    void set_record_activation_frequency(bool record) {
        record_activation_frequency = record;
        if (!record) {
            std::fill(activation_frequency.begin(), activation_frequency.end(), 0);
        }
    }

    int run_single_simulation(bool use_random_seed = true, unsigned int seed = 0) const {
        std::mt19937 rng = create_rng(use_random_seed, seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        if (record_activated) {
            last_activated_nodes.clear();
            return run_single_trial(rng, dist, seeds, &last_activated_nodes);
        } else {
            return run_single_trial(rng, dist, seeds, nullptr);
        }
    }

    std::set<int> get_activated_nodes() const {
        return last_activated_nodes;
    }

    std::vector<int> get_activation_frequency() const {
        return activation_frequency;
    }

    double run_monte_carlo_diffusion(int rounds,
        unsigned int seed = 0,
        bool use_multithread = false,
        int num_threads = 0) const {

        if (rounds <= 0) return 0.0;

        std::vector<unsigned int> trial_seeds = generate_trial_seeds(rounds, seed);

        if (record_activated) {
            last_activated_nodes.clear();
        }

        if (record_activation_frequency) {
            std::fill(activation_frequency.begin(), activation_frequency.end(), 0);
        }

        if (!use_multithread) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double sum = 0.0;

            for (int i = 0; i < rounds; ++i) {
                std::mt19937 rng(trial_seeds[i]);
                if (record_activated) {
                    std::set<int> trial_activated;
                    sum += run_single_trial(rng, dist, seeds, &trial_activated);
                    last_activated_nodes.insert(trial_activated.begin(), trial_activated.end());
                } else if (record_activation_frequency) {
                    sum += run_single_trial(rng, dist, seeds);
                } else {
                    sum += run_single_trial(rng, dist, seeds);
                }
            }
            return sum / rounds;
        }

        int actual_num_threads = (num_threads > 0) ? num_threads : std::thread::hardware_concurrency();
        actual_num_threads = std::max(1, actual_num_threads);

        std::vector<std::thread> threads;
        std::vector<double> partial_sums(actual_num_threads, 0.0);
        std::vector<std::set<int>> partial_activated(actual_num_threads);
        std::vector<std::vector<int>> partial_frequency(actual_num_threads, std::vector<int>(num_nodes, 0));

        for (int t = 0; t < actual_num_threads; ++t) {
            threads.emplace_back([this, t, actual_num_threads, rounds, &trial_seeds, &partial_sums, &partial_activated, &partial_frequency]() {
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double local_sum = 0.0;

                for (int i = t; i < rounds; i += actual_num_threads) {
                    std::mt19937 rng(trial_seeds[i]);
                    if (this->record_activated) {
                        std::set<int> trial_activated;
                        local_sum += this->run_single_trial(rng, dist, seeds, &trial_activated);
                        partial_activated[t].insert(trial_activated.begin(), trial_activated.end());
                    } else if (this->record_activation_frequency) {
                        std::vector<double> threshold(this->num_nodes);
                        std::vector<char> activated(this->num_nodes, 0);
                        std::vector<double> influence(this->num_nodes, 0.0);
                        std::vector<int> q;
                        q.reserve(this->num_nodes);
                        for (int j = 0; j < this->num_nodes; ++j) {
                            threshold[j] = this->theta_l + dist(rng) * (this->theta_h - this->theta_l);
                        }
                        int count = 0;
                        for (int s : this->seeds) {
                            activated[s] = 1;
                            q.push_back(s);
                            count++;
                            partial_frequency[t][s]++;
                        }
                        size_t front = 0;
                        while (front < q.size()) {
                            int u = q[front++];
                            const auto& neighbors = this->graph->out_neighbors(u);
                            for (const auto& edge : neighbors) {
                                int v = edge.to;
                                double w = edge.weight;
                                if (activated[v]) continue;
                                influence[v] += w;
                                if (influence[v] >= threshold[v]) {
                                    activated[v] = 1;
                                    q.push_back(v);
                                    count++;
                                    partial_frequency[t][v]++;
                                }
                            }
                        }
                        local_sum += count;
                    } else {
                        local_sum += this->run_single_trial(rng, dist, seeds);
                    }
                }
                partial_sums[t] = local_sum;
            });
        }

        for (auto& th : threads) th.join();

        double total = 0.0;
        for (double val : partial_sums) {
            total += val;
        }
        
        if (record_activated) {
            for (const auto& partial : partial_activated) {
                last_activated_nodes.insert(partial.begin(), partial.end());
            }
        }
        
        if (record_activation_frequency) {
            for (const auto& partial : partial_frequency) {
                for (int i = 0; i < num_nodes; ++i) {
                    activation_frequency[i] += partial[i];
                }
            }
        }
        
        return total / rounds;
    }
};

}
