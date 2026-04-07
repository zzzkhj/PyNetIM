#pragma once
#include "common.h"

namespace pynetim {

class DiffusionModelBase {
protected:
    std::set<int> seeds;
    std::shared_ptr<Graph> graph;
    int num_nodes;
    bool record_activated;
    mutable std::set<int> last_activated_nodes;
    bool record_activation_frequency;
    mutable std::vector<int> activation_frequency;

    virtual int run_single_trial(
        std::mt19937& rng,
        std::uniform_real_distribution<double>& dist,
        const std::set<int>& trial_seeds,
        std::set<int>* activated_nodes = nullptr,
        std::vector<int>* activation_count = nullptr) const = 0;

public:
    DiffusionModelBase(
        std::shared_ptr<Graph> graph_ptr,
        const std::set<int>& seeds,
        bool record_activated = false,
        bool record_activation_frequency = false)
        : seeds(seeds), graph(graph_ptr), num_nodes(graph_ptr->num_nodes),
          record_activated(record_activated), record_activation_frequency(record_activation_frequency) {
        activation_frequency.resize(num_nodes, 0);
    }

    virtual ~DiffusionModelBase() = default;

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

    int run_single_simulation(bool use_random_seed = true, unsigned int seed = 0) {
        std::mt19937 rng(use_random_seed ? std::random_device{}() : seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        if (record_activated) {
            return run_single_trial(rng, dist, seeds, &last_activated_nodes);
        } else if (record_activation_frequency) {
            std::vector<int> count_per_node(num_nodes, 0);
            int result = run_single_trial(rng, dist, seeds, nullptr, &count_per_node);
            for (int i = 0; i < num_nodes; ++i) {
                activation_frequency[i] += count_per_node[i];
            }
            return result;
        } else {
            return run_single_trial(rng, dist, seeds);
        }
    }

    std::set<int> get_activated_nodes() const {
        return last_activated_nodes;
    }

    std::vector<int> get_activation_frequency() const {
        return activation_frequency;
    }

    double run_monte_carlo_diffusion(int rounds, bool use_random_seed = true, unsigned int seed = 0,
                                      bool use_multithread = false, int num_threads = 0) {
        if (rounds <= 0) return 0.0;

        std::vector<unsigned int> trial_seeds(rounds);
        if (use_random_seed) {
            std::mt19937 seed_rng(std::random_device{}());
            for (int i = 0; i < rounds; ++i) {
                trial_seeds[i] = seed_rng();
            }
        } else {
            std::mt19937 seed_rng(seed);
            for (int i = 0; i < rounds; ++i) {
                trial_seeds[i] = seed_rng();
            }
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
                    std::vector<int> count_per_node(num_nodes, 0);
                    sum += run_single_trial(rng, dist, seeds, nullptr, &count_per_node);
                    for (int j = 0; j < num_nodes; ++j) {
                        activation_frequency[j] += count_per_node[j];
                    }
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
            threads.emplace_back([this, t, actual_num_threads, rounds, &trial_seeds,
                                  &partial_sums, &partial_activated, &partial_frequency]() {
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double local_sum = 0.0;

                for (int i = t; i < rounds; i += actual_num_threads) {
                    std::mt19937 rng(trial_seeds[i]);
                    if (this->record_activated) {
                        std::set<int> trial_activated;
                        local_sum += this->run_single_trial(rng, dist, seeds, &trial_activated);
                        partial_activated[t].insert(trial_activated.begin(), trial_activated.end());
                    } else if (this->record_activation_frequency) {
                        std::vector<int> count_per_node(this->num_nodes, 0);
                        local_sum += this->run_single_trial(rng, dist, seeds, nullptr, &count_per_node);
                        for (int j = 0; j < this->num_nodes; ++j) {
                            partial_frequency[t][j] += count_per_node[j];
                        }
                    } else {
                        local_sum += this->run_single_trial(rng, dist, seeds);
                    }
                }
                partial_sums[t] = local_sum;
            });
        }

        for (auto& th : threads) th.join();

        double total = std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0);

        if (record_activated) {
            last_activated_nodes.clear();
            for (int t = 0; t < actual_num_threads; ++t) {
                last_activated_nodes.insert(partial_activated[t].begin(), partial_activated[t].end());
            }
        }

        if (record_activation_frequency) {
            for (int t = 0; t < actual_num_threads; ++t) {
                for (int j = 0; j < num_nodes; ++j) {
                    activation_frequency[j] += partial_frequency[t][j];
                }
            }
        }

        return total / rounds;
    }
};

}
