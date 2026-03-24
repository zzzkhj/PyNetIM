#pragma once
#include <iostream>
#include <vector>
#include <set>
#include <unordered_map>
#include <tuple>
#include <queue>
#include <random>
#include <thread>
#include <numeric>
#include <atomic>
#include <algorithm>
#include <memory>
#include <stack>
#include <immintrin.h>
#include <pybind11/pybind11.h>

#include "Graph.h"

namespace py = pybind11;

namespace pynetim {

template<typename T>
class ObjectPool {
private:
    std::vector<std::unique_ptr<T>> pool;
    std::stack<T*> available;
    size_t capacity;

public:
    ObjectPool(size_t initial_capacity = 16) : capacity(initial_capacity) {
        for (size_t i = 0; i < initial_capacity; ++i) {
            auto obj = std::make_unique<T>();
            available.push(obj.get());
            pool.push_back(std::move(obj));
        }
    }

    T* acquire() {
        if (available.empty()) {
            auto obj = std::make_unique<T>();
            T* ptr = obj.get();
            pool.push_back(std::move(obj));
            return ptr;
        }
        T* ptr = available.top();
        available.pop();
        return ptr;
    }

    void release(T* obj) {
        available.push(obj);
    }

    void clear() {
        available = std::stack<T*>();
    }

    size_t size() const {
        return pool.size();
    }

    size_t available_count() const {
        return available.size();
    }
};

class IndependentCascadeModel {
private:
    std::set<int> seeds;
    std::shared_ptr<Graph> graph;
    int num_nodes;
    py::object py_graph_ref;

    inline int count_activated_simd(const std::vector<char>& activated) const {
        int count = 0;
        size_t i = 0;
        const size_t vec_size = 32;
        const size_t limit = activated.size() - (activated.size() % vec_size);

        for (; i < limit; i += vec_size) {
            __m256i vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&activated[i]));
            __m256i cmp = _mm256_cmpeq_epi8(vec, _mm256_set1_epi8(1));
            int mask = _mm256_movemask_epi8(cmp);
            count += __builtin_popcount(mask);
        }

        for (; i < activated.size(); ++i) {
            count += activated[i];
        }

        return count;
    }

    int run_single_trial(std::mt19937& rng,
        std::uniform_real_distribution<double>& dist,
        const std::set<int>& trial_seeds) const {

        std::vector<char> activated(num_nodes, 0);
        std::vector<int> q;

        q.reserve(num_nodes);

        for (int s : trial_seeds) {
            activated[s] = 1;
        }

        q.assign(trial_seeds.begin(), trial_seeds.end());
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
                }
            }
        }

        return count_activated_simd(activated);
    }

public:
    IndependentCascadeModel(
        std::shared_ptr<Graph> graph_ptr,
        const std::set<int>& seeds,
        py::object py_graph = py::none())
        : seeds(seeds), graph(graph_ptr), num_nodes(graph_ptr->num_nodes), py_graph_ref(py_graph) {
    }

    void set_seeds(const std::set<int>& new_seeds) {
        seeds = new_seeds;
    }

    double run_monte_carlo_diffusion(int rounds,
        unsigned int seed = 0,
        bool use_multithread = false,
        int num_threads = 0) const {

        if (rounds <= 0) return 0.0;

        std::vector<unsigned int> trial_seeds(rounds);
        {
            std::mt19937 master_rng(seed);
            for (int i = 0; i < rounds; ++i) {
                trial_seeds[i] = master_rng();
            }
        }

        if (!use_multithread) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double sum = 0.0;

            for (int i = 0; i < rounds; ++i) {
                std::mt19937 rng(trial_seeds[i]);
                sum += run_single_trial(rng, dist, seeds);
            }
            return sum / rounds;
        }

        int actual_num_threads = (num_threads > 0) ? num_threads : std::thread::hardware_concurrency();
        actual_num_threads = std::max(1, actual_num_threads);

        std::vector<std::thread> threads;
        std::vector<double> partial_sums(actual_num_threads, 0.0);

        for (int t = 0; t < actual_num_threads; ++t) {
            threads.emplace_back([this, t, actual_num_threads, rounds, &trial_seeds, &partial_sums]() {
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double local_sum = 0.0;

                for (int i = t; i < rounds; i += actual_num_threads) {
                    std::mt19937 rng(trial_seeds[i]);
                    local_sum += this->run_single_trial(rng, dist, seeds);
                }
                partial_sums[t] = local_sum;
            });
        }

        for (auto& th : threads) th.join();

        double total = std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0);
        return total / rounds;
    }
};



class LinearThresholdModel {
private:
    std::set<int> seeds;
    std::shared_ptr<Graph> graph;
    int num_nodes;
    double theta_l;
    double theta_h;

    inline int count_activated_simd(const std::vector<char>& activated) const {
        int count = 0;
        size_t i = 0;
        const size_t vec_size = 32;
        const size_t limit = activated.size() - (activated.size() % vec_size);

        for (; i < limit; i += vec_size) {
            __m256i vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&activated[i]));
            __m256i cmp = _mm256_cmpeq_epi8(vec, _mm256_set1_epi8(1));
            int mask = _mm256_movemask_epi8(cmp);
            count += __builtin_popcount(mask);
        }

        for (; i < activated.size(); ++i) {
            count += activated[i];
        }

        return count;
    }

    int run_single_trial(std::mt19937& rng,
        std::uniform_real_distribution<double>& dist,
        const std::set<int>& trial_seeds) const {

        std::vector<double> threshold(num_nodes);
        std::vector<char> activated(num_nodes, 0);
        std::vector<double> influence(num_nodes, 0.0);
        std::vector<int> q;

        q.reserve(num_nodes);

        for (int i = 0; i < num_nodes; ++i) {
            threshold[i] = theta_l + dist(rng) * (theta_h - theta_l);
        }

        q.assign(trial_seeds.begin(), trial_seeds.end());
        for (int s : trial_seeds) {
            activated[s] = 1;
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
                }
            }
        }

        return count_activated_simd(activated);
    }

public:
    LinearThresholdModel(
        std::shared_ptr<Graph> graph_ptr,
        const std::set<int>& seeds,
        double theta_l = 0.0,
        double theta_h = 1.0)
        : seeds(seeds), graph(graph_ptr), num_nodes(graph_ptr->num_nodes),
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

    void set_seeds(const std::set<int>& new_seeds) {
        seeds = new_seeds;
    }

    double run_monte_carlo_diffusion(int rounds,
        unsigned int seed = 0,
        bool use_multithread = false,
        int num_threads = 0) const {

        if (rounds <= 0) return 0.0;

        std::vector<unsigned int> trial_seeds(rounds);
        {
            std::mt19937 master_rng(seed);
            for (int i = 0; i < rounds; ++i) {
                trial_seeds[i] = master_rng();
            }
        }

        if (!use_multithread) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double sum = 0.0;

            for (int i = 0; i < rounds; ++i) {
                std::mt19937 rng(trial_seeds[i]);
                sum += run_single_trial(rng, dist, seeds);
            }
            return sum / rounds;
        }

        int actual_num_threads = (num_threads > 0) ? num_threads : std::thread::hardware_concurrency();
        actual_num_threads = std::max(1, actual_num_threads);

        std::vector<std::thread> threads;
        std::vector<double> partial_sums(actual_num_threads, 0.0);

        for (int t = 0; t < actual_num_threads; ++t) {
            threads.emplace_back([this, t, actual_num_threads, rounds, &trial_seeds, &partial_sums]() {
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double local_sum = 0.0;

                for (int i = t; i < rounds; i += actual_num_threads) {
                    std::mt19937 rng(trial_seeds[i]);
                    local_sum += this->run_single_trial(rng, dist, seeds);
                }
                partial_sums[t] = local_sum;
            });
        }

        for (auto& th : threads) th.join();

        double total = 0.0;
        for (double val : partial_sums) {
            total += val;
        }
        return total / rounds;
    }
};

}
