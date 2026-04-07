#pragma once

#include "base_ris_algorithm.h"

namespace pynetim {

class TIMAlgorithm : public BaseRISAlgorithm {
private:
    double epsilon_;
    int    l_;

    double estimateOPT(int k) {
        if (verbose_) {
            std::cerr << "[TIM] 阶段1: 估计OPT..." << std::endl;
        }
        double eps_p = epsilon_ / (1.0 + epsilon_);

        int64_t R = static_cast<int64_t>(
            (2.0 + 2.0/3.0 * eps_p) *
            (l_ * std::log(graph_->num_nodes)
             + logCnk(graph_->num_nodes, k)
             + std::log(2.0)) /
            (eps_p * eps_p)
        );

        if (verbose_) {
            std::cerr << "[TIM]   采样 " << R << " 个RR集合..." << std::endl;
        }
        buildHyperGraphR(R);
        double ept = buildSeedSet(k);
        double OPT = ept * graph_->num_nodes;
        if (verbose_) {
            std::cerr << "[TIM]   估计 OPT = " << OPT
                      << " (ept=" << ept << ")" << std::endl;
        }
        return OPT;
    }

public:
    TIMAlgorithm(std::shared_ptr<Graph> graph, const std::string& model,
                 double epsilon = 0.5, int l = 1,
                 std::optional<int> seed = std::nullopt,
                 bool verbose = false)
        : BaseRISAlgorithm(graph, model, seed, verbose), epsilon_(epsilon), l_(l) {
        if (epsilon_ <= 0 || epsilon_ >= 1)
            throw std::invalid_argument("epsilon 必须在 (0, 1) 范围内");
    }

    std::set<int> run(int k) {
        if (k <= 0) return {};
        hyperGT_.clear();
        hyperG_.clear();
        hyperG_.resize(graph_->num_nodes);
        seedSet_.clear();

        double OPT_prime = estimateOPT(k);

        if (verbose_) {
            std::cerr << "[TIM] 阶段2: 最终采样与选择..." << std::endl;
        }
        int64_t R = static_cast<int64_t>(
            (2.0 + 2.0/3.0 * epsilon_) *
            graph_->num_nodes *
            (l_ * std::log(graph_->num_nodes)
             + logCnk(graph_->num_nodes, k)
             + std::log(2.0)) /
            (OPT_prime * epsilon_ * epsilon_)
        );

        if (verbose_) {
            std::cerr << "[TIM]   采样 " << R << " 个RR集合..." << std::endl;
        }
        buildHyperGraphR(R);
        buildSeedSet(k);
        if (verbose_) {
            std::cerr << "[TIM]   总RR集合数: "
                      << hyperGT_.size() << std::endl;
        }
        return seedSet_;
    }
};

class TIMPlusAlgorithm : public BaseRISAlgorithm {
private:
    double epsilon_;
    int    l_;

    double estimateOPTPlus(int k) {
        if (verbose_) {
            std::cerr << "[TIM+] 阶段1: 自适应估计OPT..." << std::endl;
        }
        double eps_p = epsilon_ * std::sqrt(2.0);

        int max_iter = static_cast<int>(std::floor(std::log2(graph_->num_nodes)));
        int64_t prev_ci = 0;

        for (int x = 1; x <= max_iter; x++) {
            int64_t ci = static_cast<int64_t>(
                (6.0 * l_ * std::log(graph_->num_nodes)
                 + 6.0 * std::log(std::log2(graph_->num_nodes))) *
                std::pow(2.0, x) / (eps_p * eps_p)
            );

            if (verbose_) {
                std::cerr << "[TIM+]   第 " << x << " 轮迭代"
                          << ": 额外采样 " << (ci - prev_ci)
                          << " 个RR集合 (总计: " << ci << ")..." << std::endl;
            }
            buildHyperGraphR(ci);
            prev_ci = ci;

            double ept = buildSeedSet(k);
            if (ept > 1.0 / std::pow(2.0, x)) {
                double OPT = ept * graph_->num_nodes;
                if (verbose_) {
                    std::cerr << "[TIM+]   找到 OPT = " << OPT
                              << " (ept=" << ept << ")" << std::endl;
                }
                return OPT;
            }
        }

        if (verbose_) {
            std::cerr << "[TIM+]   使用备用 OPT = n" << std::endl;
        }
        return static_cast<double>(graph_->num_nodes);
    }

public:
    TIMPlusAlgorithm(std::shared_ptr<Graph> graph, const std::string& model,
                     double epsilon = 0.5, int l = 1,
                     std::optional<int> seed = std::nullopt,
                     bool verbose = false)
        : BaseRISAlgorithm(graph, model, seed, verbose), epsilon_(epsilon), l_(l) {
        if (epsilon_ <= 0 || epsilon_ >= 1)
            throw std::invalid_argument("epsilon 必须在 (0, 1) 范围内");
    }

    std::set<int> run(int k) {
        if (k <= 0) return {};
        hyperGT_.clear();
        hyperG_.clear();
        hyperG_.resize(graph_->num_nodes);
        seedSet_.clear();

        double OPT_prime = estimateOPTPlus(k);

        if (verbose_) {
            std::cerr << "[TIM+] 阶段2: 最终采样与选择..." << std::endl;
        }
        int64_t R = static_cast<int64_t>(
            (2.0 + 2.0/3.0 * epsilon_) *
            graph_->num_nodes *
            (l_ * std::log(graph_->num_nodes)
             + logCnk(graph_->num_nodes, k)
             + std::log(2.0)) /
            (OPT_prime * epsilon_ * epsilon_)
        );

        if (verbose_) {
            std::cerr << "[TIM+]   采样 " << R << " 个RR集合..." << std::endl;
        }
        buildHyperGraphR(R);
        buildSeedSet(k);
        if (verbose_) {
            std::cerr << "[TIM+]   总RR集合数: "
                      << hyperGT_.size() << std::endl;
        }
        return seedSet_;
    }
};

} // namespace pynetim
