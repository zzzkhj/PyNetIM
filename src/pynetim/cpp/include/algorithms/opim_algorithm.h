#pragma once

#include <limits>
#include <algorithm>
#include "base_ris_algorithm.h"

namespace pynetim {

class OPIMAlgorithm : public BaseRISAlgorithm {
protected:
    std::vector<std::vector<int>> hyperGT_vldt_; 
    std::vector<std::vector<int>> hyperG_vldt_;  

    int64_t numRRsets_ = 0;
    double approximation_ = 0.0;
    double influence_ = 0.0;

    void buildHyperGraphRVldt(int64_t R) {
        int prev_size = static_cast<int>(hyperGT_vldt_.size());
        if (R <= prev_size) return;
        
        hyperGT_vldt_.resize(R);
        hyperG_vldt_.resize(graph_->num_nodes);
        
        for (int i = prev_size; i < static_cast<int>(R); i++) {
            hyperGT_vldt_[i] = (model_ == "IC") ? sampleRRSetIC() : sampleRRSetLT();
            for (int node : hyperGT_vldt_[i]) {
                hyperG_vldt_[node].push_back(i);
            }
        }
    }

    double greedySelection(int k, int64_t numRRsets) {
        std::vector<int> coverage(graph_->num_nodes, 0);
        for (int i = 0; i < graph_->num_nodes; i++) {
            coverage[i] = static_cast<int>(hyperG_[i].size());
        }

        std::priority_queue<std::pair<int, int>> heap;
        for (int i = 0; i < graph_->num_nodes; i++) {
            if (coverage[i] > 0) {
                heap.push({coverage[i], i});
            }
        }

        std::vector<bool> edgeMark(numRRsets, false);
        seedSet_.clear();
        int64_t totalCov = 0;

        while (static_cast<int>(seedSet_.size()) < k && !heap.empty()) {
            auto [cov, node] = heap.top();
            heap.pop();

            if (cov != coverage[node]) {
                heap.push({coverage[node], node});
                continue;
            }

            totalCov += coverage[node];
            seedSet_.insert(node);
            coverage[node] = 0;

            for (int rrIdx : hyperG_[node]) {
                if (edgeMark[rrIdx]) continue;
                edgeMark[rrIdx] = true;
                for (int n : hyperGT_[rrIdx]) {
                    if (coverage[n] > 0) {
                        coverage[n]--;
                    }
                }
            }
        }

        return static_cast<double>(totalCov) * graph_->num_nodes / numRRsets;
    }

    double estimateInfluence(const std::set<int>& seeds,
                             std::vector<std::vector<int>>& hyperG,
                             int64_t numRRsets) {
        std::vector<bool> covered(numRRsets, false);
        for (int seed : seeds) {
            for (int rrIdx : hyperG[seed]) {
                covered[rrIdx] = true;
            }
        }
        int count = static_cast<int>(std::count(covered.begin(), covered.end(), true));
        return static_cast<double>(count) * graph_->num_nodes / numRRsets;
    }

    double computeLowerBound(double estimate, int64_t numSamples, double delta) {
        double error = std::sqrt(std::log(2.0 / delta) / (2.0 * numSamples));
        return estimate * (1.0 - error);
    }

    double computeUpperBound(double estimate, int64_t numSamples, double delta) {
        double error = std::sqrt(std::log(2.0 / delta) / (2.0 * numSamples));
        return estimate * (1.0 + error);
    }

public:
    OPIMAlgorithm(std::shared_ptr<Graph> graph,
                  const std::string& model,
                  std::optional<int> seed = std::nullopt,
                  bool verbose = false)
        : BaseRISAlgorithm(graph, model, seed, verbose) {}

    virtual ~OPIMAlgorithm() = default;

    std::set<int> run(int k, int64_t numRRsets, double delta = -1.0) {
        if (k <= 0 || numRRsets <= 0) return {};

        if (delta < 0) {
            delta = 1.0 / graph_->num_nodes;
        }

        int64_t trainSize = numRRsets;
        int64_t validSize = numRRsets;

        if (verbose_) {
            std::cerr << "[OPIM] 构建训练集: " << trainSize << " 个RR集合..." << std::endl;
        }
        hyperGT_.clear();
        hyperG_.clear();
        hyperG_.resize(graph_->num_nodes);
        buildHyperGraphR(trainSize);
        numRRsets_ = static_cast<int64_t>(hyperGT_.size());

        if (verbose_) {
            std::cerr << "[OPIM] 构建验证集: " << validSize << " 个RR集合..." << std::endl;
        }
        hyperGT_vldt_.clear();
        hyperG_vldt_.clear();
        hyperG_vldt_.resize(graph_->num_nodes);
        buildHyperGraphRVldt(validSize);
        int64_t numRRsetsVldt = static_cast<int64_t>(hyperGT_vldt_.size());

        if (verbose_) {
            std::cerr << "[OPIM] 执行贪心选择..." << std::endl;
        }
        double trainInf = greedySelection(k, numRRsets_);

        double validInf = estimateInfluence(seedSet_, hyperG_vldt_, numRRsetsVldt);

        double lowerBound = computeLowerBound(validInf, numRRsetsVldt, delta);
        double upperBound = computeUpperBound(trainInf, numRRsets_, delta);

        approximation_ = (upperBound > 0) ? lowerBound / upperBound : 0.0;
        influence_ = validInf;

        if (verbose_) {
            std::cerr << "[OPIM] 训练集影响力: " << trainInf << std::endl;
            std::cerr << "[OPIM] 验证集影响力: " << validInf << std::endl;
            std::cerr << "[OPIM] 下界: " << lowerBound << std::endl;
            std::cerr << "[OPIM] 上界: " << upperBound << std::endl;
            std::cerr << "[OPIM] 近似比: " << approximation_ << std::endl;
        }

        return seedSet_;
    }

    double getApproximation() const { return approximation_; }
    double getInfluence() const { return influence_; }
};

class OPIMCAlgorithm : public OPIMAlgorithm {
public:
    OPIMCAlgorithm(std::shared_ptr<Graph> graph,
                   const std::string& model,
                   std::optional<int> seed = std::nullopt,
                   bool verbose = false)
        : OPIMAlgorithm(graph, model, seed, verbose) {}

    std::set<int> run(int k, double epsilon, double delta = -1.0) {
        if (k <= 0) return {};

        if (delta < 0) {
            delta = 1.0 / graph_->num_nodes;
        }

        const double e = std::exp(1.0);
        const double targetApprox = 1.0 - 1.0 / e - epsilon;

        double alpha = std::sqrt(std::log(6.0 / delta));
        double beta = std::sqrt((1.0 - 1.0/e) * (logCnk(graph_->num_nodes, k) + std::log(6.0 / delta)));

        int64_t numRbase = static_cast<int64_t>(2.0 * std::pow((1.0 - 1.0/e) * alpha + beta, 2));
        int64_t maxNumR = static_cast<int64_t>(2.0 * graph_->num_nodes * 
            std::pow((1.0 - 1.0/e) * alpha + beta, 2) / k / (epsilon * epsilon)) + 1;

        int numIter = static_cast<int>(std::log2(static_cast<double>(maxNumR) / numRbase)) + 1;

        double iterDelta = delta / (3.0 * numIter);

        if (verbose_) {
            std::cerr << "[OPIM-C] 基础RR数: " << numRbase << ", 最大RR数: " << maxNumR 
                      << ", 迭代次数: " << numIter << ", 目标近似比: " << targetApprox << std::endl;
        }

        hyperGT_.clear();
        hyperG_.clear();
        hyperG_.resize(graph_->num_nodes);
        hyperGT_vldt_.clear();
        hyperG_vldt_.clear();
        hyperG_vldt_.resize(graph_->num_nodes);

        for (int idx = 0; idx < numIter; idx++) {
            int64_t numR = numRbase << idx;

            if (verbose_) {
                std::cerr << "[OPIM-C] 第 " << (idx + 1) << "/" << numIter 
                          << " 轮迭代: 采样 " << numR << " 个RR集合..." << std::endl;
            }

            buildHyperGraphR(numR);
            buildHyperGraphRVldt(numR);

            numRRsets_ = static_cast<int64_t>(hyperGT_.size());
            int64_t numRRsetsVldt = static_cast<int64_t>(hyperGT_vldt_.size());

            double trainInf = greedySelection(k, numRRsets_);
            double validInf = estimateInfluence(seedSet_, hyperG_vldt_, numRRsetsVldt);

            double lowerBound = computeLowerBound(validInf, numRRsetsVldt, iterDelta);
            double upperBound = computeUpperBound(trainInf, numRRsets_, iterDelta);

            double approxOPIMC = (upperBound > 0) ? lowerBound / upperBound : 0.0;

            if (verbose_) {
                std::cerr << "[OPIM-C]   R1大小: " << numRRsets_ 
                          << ", R2大小: " << numRRsetsVldt
                          << ", 训练影响力: " << trainInf
                          << ", 验证影响力: " << validInf
                          << ", 近似比: " << approxOPIMC 
                          << " (目标: " << targetApprox << ")" << std::endl;
            }

            if (approxOPIMC >= targetApprox) {
                approximation_ = approxOPIMC;
                influence_ = validInf;
                if (verbose_) {
                    std::cerr << "[OPIM-C] 收敛! 近似比: " << approximation_ 
                              << ", 影响力: " << influence_ << std::endl;
                }
                return seedSet_;
            }
        }

        approximation_ = 0.0;
        influence_ = estimateInfluence(seedSet_, hyperG_vldt_, static_cast<int64_t>(hyperGT_vldt_.size()));
        if (verbose_) {
            std::cerr << "[OPIM-C] 未收敛。最终影响力: " << influence_ << std::endl;
        }
        return seedSet_;
    }
};

} // namespace pynetim
