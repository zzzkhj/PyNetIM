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

#include "../graph/Graph.h"
//#include "utils.h"

class IndependentCascadeModel {
private:
    std::set<int> seeds;
    Graph graph;

    // 只读邻接表（线程安全）
    //std::unordered_map<int, std::vector<std::pair<int, double>>> adj_list;

    // 节点数量（用于 vector<char> activated）
    int num_nodes;

    //std::map

private:
    // 单次 IC 扩散（一个 Monte Carlo trial）
    int run_single_trial(std::mt19937& rng,
        std::uniform_real_distribution<double>& dist) const {

        // 使用 vector<char> 代替 unordered_set，假设节点 ID 连续 0~num_nodes-1
        std::vector<char> activated(num_nodes, 0);

        // 初始化种子节点
        for (int s : seeds) {
            activated[s] = 1;
        }

        // 使用 vector 作为队列，手动索引 BFS
        std::vector<int> q(seeds.begin(), seeds.end());
        size_t front = 0;

        while (front < q.size()) {
            int u = q[front++];
            const auto& neighbors = graph.out_neighbors(u);  

            for (const auto& v : neighbors) {
                //int v = edge.first;
                double w = graph.edges.at({ u, v });

                if (activated[v] == 0 && dist(rng) < w) {
                    activated[v] = 1;
                    q.push_back(v);
                }
            }
        }

        // 统计激活节点数
         return std::reduce(activated.begin(), activated.end(), 0);
    }

public:
    IndependentCascadeModel(const std::set<int>& seeds,
        const Graph& graph)
        : seeds(seeds), graph(graph), num_nodes(graph.num_nodes) {
    }

    void set_seeds(const std::set<int>& new_seeds) {
        seeds = new_seeds;
    }

    // =========================
    // Monte Carlo Diffusion
    // =========================
    double run_monte_carlo_diffusion(int rounds,
        unsigned int seed = 0,
        bool use_multithread = false) const {

        if (rounds <= 0) return 0.0;

        // 预先生成所有 trial 的随机种子（保证单/多线程结果完全一致）
        std::vector<unsigned int> trial_seeds(rounds);
        {
            std::mt19937 master_rng(seed);
            for (int i = 0; i < rounds; ++i) {
                trial_seeds[i] = master_rng();
            }
        }

        // ---------- 单线程 ----------
        if (!use_multithread) {  
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double sum = 0.0;

            for (int i = 0; i < rounds; ++i) {
                std::mt19937 rng(trial_seeds[i]);
                sum += run_single_trial(rng, dist);
            }
            return sum / rounds;
        }

        // ---------- 多线程 ----------
        int num_threads = std::thread::hardware_concurrency();
        num_threads = std::max(1, num_threads);
        // 如果 rounds 不足以充分利用线程，减少线程数
      /*  if (rounds < num_threads * 100) {
            num_threads = std::max(1, rounds / 100);
        }*/

        std::vector<std::thread> threads;
        std::vector<double> partial_sums(num_threads, 0.0);

        auto worker = [&](int tid) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double local_sum = 0.0;

            for (int i = tid; i < rounds; i += num_threads) {
                std::mt19937 rng(trial_seeds[i]);
                local_sum += run_single_trial(rng, dist);
            }
            partial_sums[tid] = local_sum;
            };

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(worker, t);
        }

        for (auto& th : threads) th.join();

        double total = std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0);
        return total / rounds;
    }
};




class LinearThresholdModel {
private:
    std::set<int> seeds;
    Graph graph;
    int num_nodes;
    double theta_l = 0.0;
    double theta_h = 1.0;

    // 单次 LT 扩散（一个 Monte Carlo trial）
    int run_single_trial(std::mt19937& rng,
        std::uniform_real_distribution<double>& dist) const {
        // 每个节点的随机阈值 θ_v ∈ [theta_l,theta_h)
        std::vector<double> threshold(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            threshold[i] = theta_l + dist(rng) * (theta_h - theta_l);
        }

        // 激活状态
        std::vector<char> activated(num_nodes, 0);

        // 当前已激活邻居对每个节点的影响累计
        std::vector<double> influence(num_nodes, 0.0);

        // 初始化种子
        std::vector<int> q(seeds.begin(), seeds.end());
        for (int s : seeds) {
            activated[s] = 1;
        }

        size_t front = 0;
        while (front < q.size()) {
            int u = q[front++];
            // 将 u 的影响传播到所有出邻居 v
            const auto& neighbors = graph.out_neighbors(u);
            for (int v : neighbors) {
                if (activated[v]) continue;  // 已激活的不再处理

                double w = graph.edges.at({ u, v });
                //std::cout << v << " " << influence[v] << std::endl;
                influence[v] += w;

                // 如果累计影响超过阈值，则激活 v
                if (influence[v] >= threshold[v]) {
                    //std::cout << u << " " << v << influence[v] << " " << threshold[v] << std::endl;
                    activated[v] = 1;
                    q.push_back(v);
                }
            }
        }

        // 统计激活节点数
        return std::reduce(activated.begin(), activated.end(), 0);
    }

public:
    LinearThresholdModel(const std::set<int>& seeds,
        const Graph& graph,
        double theta_l = 0.0,
        double theta_h = 1.0)
        : seeds(seeds), graph(graph), num_nodes(graph.num_nodes)
    {
        // 校验 theta_l 和 theta_h
        if (theta_l < 0.0 || theta_l > 1.0) {
            throw std::invalid_argument("theta_l must be in [0,1]");
        }
        if (theta_h < 0.0 || theta_h > 1.0) {
            throw std::invalid_argument("theta_h must be in [0,1]");
        }
        if (theta_l > theta_h) {
            throw std::invalid_argument("theta_l cannot be greater than theta_h");
        }

        // 成员赋值
        this->theta_l = theta_l;
        this->theta_h = theta_h;
    }


    void set_seeds(const std::set<int>& new_seeds) {
        seeds = new_seeds;
    }

    // =========================
    // Monte Carlo Diffusion
    // =========================
    double run_monte_carlo_diffusion(int rounds,
        unsigned int seed = 0,
        bool use_multithread = false) const {

        if (rounds <= 0) return 0.0;

        // 预先生成所有 trial 的随机种子（保证单/多线程结果完全一致）
        std::vector<unsigned int> trial_seeds(rounds);
        {
            std::mt19937 master_rng(seed);
            for (int i = 0; i < rounds; ++i) {
                trial_seeds[i] = master_rng();
            }
        }

        // ---------- 单线程 ----------
        if (!use_multithread) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double sum = 0.0;

            for (int i = 0; i < rounds; ++i) {
                std::mt19937 rng(trial_seeds[i]);
                sum += run_single_trial(rng, dist);
            }
            return sum / rounds;
        }

        // ---------- 多线程 ----------
        int num_threads = std::thread::hardware_concurrency();
        num_threads = std::max(1, num_threads);
        // 如果 rounds 不足以充分利用线程，减少线程数
      /*  if (rounds < num_threads * 100) {
            num_threads = std::max(1, rounds / 100);
        }*/

        std::vector<std::thread> threads;
        std::vector<double> partial_sums(num_threads, 0.0);

        auto worker = [&](int tid) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double local_sum = 0.0;

            for (int i = tid; i < rounds; i += num_threads) {
                std::mt19937 rng(trial_seeds[i]);
                local_sum += run_single_trial(rng, dist);
            }
            partial_sums[tid] = local_sum;
            };

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(worker, t);
        }

        for (auto& th : threads) th.join();

        double total = std::reduce(partial_sums.begin(), partial_sums.end(), 0.0);
        return total / rounds;
    }
};