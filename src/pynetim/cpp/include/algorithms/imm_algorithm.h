#pragma once

#include "base_ris_algorithm.h"

namespace pynetim {

class IMMAlgorithm : public BaseRISAlgorithm {
private:
    double epsilon_;
    int    l_;

    double step1(int k) {
        if (verbose_) {
            std::cerr << "[IMM] 阶段1: 估计OPT下界..." << std::endl;
        }
        double eps_p = epsilon_ * std::sqrt(2.0);

        int64_t prev_ci = 0;
        for (int x = 1; ; x++) {
            int64_t ci = static_cast<int64_t>(
                (2.0 + 2.0/3.0 * eps_p) *
                (l_ * std::log(graph_->num_nodes)
                 + logCnk(graph_->num_nodes, k)
                 + std::log(std::log2(graph_->num_nodes))) *
                std::pow(2.0, x) / (eps_p * eps_p)
            );

            if (verbose_) {
                std::cerr << "[IMM]   第 " << x << " 轮迭代"
                          << ": 额外采样 " << (ci - prev_ci)
                          << " 个RR集合 (总计: " << ci << ")..." << std::endl;
            }
            buildHyperGraphR(ci);
            prev_ci = ci;

            double ept = buildSeedSet(k);
            if (ept > 1.0 / std::pow(2.0, x)) {
                double OPT_prime = ept * graph_->num_nodes;
                if (verbose_) {
                    std::cerr << "[IMM]   找到 OPT' = " << OPT_prime
                              << " (ept=" << ept << ")" << std::endl;
                }
                return OPT_prime;
            }
        }
    }

    void step2(int k, double OPT_prime) {
        if (verbose_) {
            std::cerr << "[IMM] 阶段2: 最终采样与选择..." << std::endl;
        }
        const double e = std::exp(1.0);

        double alpha = std::sqrt(l_ * std::log(graph_->num_nodes) + std::log(2.0));
        double beta  = std::sqrt((1.0 - 1.0/e) *
                                 (logCnk(graph_->num_nodes, k)
                                  + l_ * std::log(graph_->num_nodes)
                                  + std::log(2.0)));

        int64_t R = static_cast<int64_t>(
            2.0 * graph_->num_nodes *
            std::pow((1.0 - 1.0/e) * alpha + beta, 2) /
            (OPT_prime * epsilon_ * epsilon_)
        );

        if (verbose_) {
            std::cerr << "[IMM]   采样 " << R << " 个RR集合..." << std::endl;
        }
        buildHyperGraphR(R);
        buildSeedSet(k);
        if (verbose_) {
            std::cerr << "[IMM]   总RR集合数: "
                      << hyperGT_.size() << std::endl;
        }
    }

public:
    IMMAlgorithm(std::shared_ptr<Graph> graph, const std::string& model,
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
        double OPT_prime = step1(k);
        step2(k, OPT_prime);
        return seedSet_;
    }
};

} // namespace pynetim
