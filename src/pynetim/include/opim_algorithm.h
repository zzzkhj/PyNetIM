#pragma once

#include <vector>
#include <queue>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>
#include <iostream>
#include <set>
#include <optional>
#include <limits>
#include "ris_algorithm.h"

namespace pynetim {

class OPIMAlgorithm : public BaseRISAlgorithm {
protected:
    std::vector<std::vector<int>> hyperGT_vldt_; 
    std::vector<std::vector<int>> hyperG_vldt_;  

    double boundLast_ = std::numeric_limits<double>::max();
    double boundMin_ = std::numeric_limits<double>::max();
    int64_t numRRsets_ = 0;
    double approximation_ = 0.0;
    double influence_ = 0.0;

    void buildHyperGraphRVldt(int64_t R) {
        hyperGT_vldt_.resize(R);
        hyperG_vldt_.assign(graph_->num_nodes, std::vector<int>());

        for (int64_t i = 0; i < R; i++) {
            hyperGT_vldt_[i] = (model_ == "IC") ? sampleRRSetIC() : sampleRRSetLT();
            for (int node : hyperGT_vldt_[i]) {
                hyperG_vldt_[node].push_back(static_cast<int>(i));
            }
        }
    }

    double maxCoverLazy(int k, int mode,
                        std::vector<std::vector<int>>& hyperGT,
                        std::vector<std::vector<int>>& hyperG,
                        int64_t numRRsets) {
        boundLast_ = std::numeric_limits<double>::max();
        boundMin_ = std::numeric_limits<double>::max();

        std::vector<int> coverage(graph_->num_nodes, 0);
        int maxDeg = 0;
        for (int i = 0; i < graph_->num_nodes; i++) {
            coverage[i] = static_cast<int>(hyperG[i].size());
            if (coverage[i] > maxDeg) maxDeg = coverage[i];
        }

        std::vector<std::vector<int>> degMap(maxDeg + 1);
        for (int i = 0; i < graph_->num_nodes; i++) {
            if (coverage[i] > 0) {
                degMap[coverage[i]].push_back(i);
            }
        }

        std::vector<bool> edgeMark(numRRsets, false);
        seedSet_.clear();
        int64_t sumInf = 0;

        for (int deg = maxDeg; deg > 0; deg--) {
            auto& vecNode = degMap[deg];
            for (size_t idx = 0; idx < vecNode.size(); idx++) {
                int argmaxIdx = vecNode[idx];
                int currDeg = coverage[argmaxIdx];

                if (deg > currDeg) {
                    degMap[currDeg].push_back(argmaxIdx);
                    continue;
                }

                if (mode == 2 || (mode == 1 && static_cast<int>(seedSet_.size()) == k)) {
                    int topk = k;
                    int degBound = deg;
                    std::vector<int> vecBound(k, 0);

                    int idxBound = static_cast<int>(idx) + 1;
                    while (topk > 0 && idxBound > 0) {
                        idxBound--;
                        topk--;
                        vecBound[topk] = coverage[degMap[degBound][idxBound]];
                    }
                    while (topk > 0 && --degBound > 0) {
                        idxBound = static_cast<int>(degMap[degBound].size());
                        while (topk > 0 && idxBound > 0) {
                            idxBound--;
                            topk--;
                            vecBound[topk] = coverage[degMap[degBound][idxBound]];
                        }
                    }

                    std::make_heap(vecBound.begin(), vecBound.end(), std::greater<int>());

                    bool flag = (topk == 0);
                    while (flag && idxBound > 0) {
                        idxBound--;
                        int currDegBound = coverage[degMap[degBound][idxBound]];
                        if (vecBound.front() >= degBound) {
                            flag = false;
                        } else if (vecBound.front() < currDegBound) {
                            std::pop_heap(vecBound.begin(), vecBound.end(), std::greater<int>());
                            vecBound.back() = currDegBound;
                            std::push_heap(vecBound.begin(), vecBound.end(), std::greater<int>());
                        }
                    }
                    while (flag && --degBound > 0) {
                        idxBound = static_cast<int>(degMap[degBound].size());
                        while (flag && idxBound > 0) {
                            idxBound--;
                            int currDegBound = coverage[degMap[degBound][idxBound]];
                            if (vecBound.front() >= degBound) {
                                flag = false;
                            } else if (vecBound.front() < currDegBound) {
                                std::pop_heap(vecBound.begin(), vecBound.end(), std::greater<int>());
                                vecBound.back() = currDegBound;
                                std::push_heap(vecBound.begin(), vecBound.end(), std::greater<int>());
                            }
                        }
                    }

                    int64_t boundSum = 0;
                    for (int v : vecBound) boundSum += v;
                    boundLast_ = static_cast<double>(boundSum + sumInf) * graph_->num_nodes / numRRsets;
                    if (boundMin_ > boundLast_) boundMin_ = boundLast_;
                }

                if (static_cast<int>(seedSet_.size()) >= k) {
                    double finalInf = static_cast<double>(sumInf) * graph_->num_nodes / numRRsets;
                    if (verbose_) {
                        std::cerr << "  >>>[greedy-lazy] influence: " << finalInf
                                  << ", min-bound: " << boundMin_
                                  << ", last-bound: " << boundLast_ << std::endl;
                    }
                    return finalInf;
                }

                sumInf += currDeg;
                seedSet_.insert(argmaxIdx);
                coverage[argmaxIdx] = 0;

                for (int edgeIdx : hyperG[argmaxIdx]) {
                    if (edgeMark[edgeIdx]) continue;
                    edgeMark[edgeIdx] = true;
                    for (int nodeIdx : hyperGT[edgeIdx]) {
                        if (coverage[nodeIdx] == 0) continue;
                        coverage[nodeIdx]--;
                    }
                }
            }
            degMap.pop_back();
        }
        return static_cast<double>(graph_->num_nodes);
    }

    double selfInfCal(const std::set<int>& seeds,
                      std::vector<std::vector<int>>& hyperG,
                      int64_t numRRsets) {
        std::vector<bool> vecBoolVst(numRRsets, false);
        for (int seed : seeds) {
            for (int rrIdx : hyperG[seed]) {
                vecBoolVst[rrIdx] = true;
            }
        }
        int count = static_cast<int>(std::count(vecBoolVst.begin(), vecBoolVst.end(), true));
        return static_cast<double>(count) * graph_->num_nodes / numRRsets;
    }

public:
    OPIMAlgorithm(std::shared_ptr<Graph> graph,
                  const std::string& model,
                  std::optional<int> seed = std::nullopt,
                  bool verbose = false)
        : BaseRISAlgorithm(graph, model, seed, verbose) {}

    virtual ~OPIMAlgorithm() = default;

    std::set<int> run(int k, int64_t numRRsets, double delta = -1.0, int mode = 2) {
        if (k <= 0 || numRRsets <= 0) return {};

        if (delta < 0) {
            delta = 1.0 / graph_->num_nodes;
        }

        const double e = std::exp(1.0);
        const double approx = 1.0 - 1.0 / e;
        const double a1 = std::log(2.0 / delta);
        const double a2 = std::log(2.0 / delta);

        if (verbose_) {
            std::cerr << "[OPIM] Building " << numRRsets/2 << " RR sets for R1..." << std::endl;
        }
        hyperGT_.clear();
        hyperG_.clear();
        hyperG_.resize(graph_->num_nodes);
        buildHyperGraphR(numRRsets / 2);

        if (verbose_) {
            std::cerr << "[OPIM] Building " << numRRsets/2 << " RR sets for R2..." << std::endl;
        }
        buildHyperGraphRVldt(numRRsets / 2);

        numRRsets_ = static_cast<int64_t>(hyperGT_.size());

        if (verbose_) {
            std::cerr << "[OPIM] Running greedy selection..." << std::endl;
        }
        double infSelf = maxCoverLazy(k, mode, hyperGT_, hyperG_, numRRsets_);

        double infVldt = selfInfCal(seedSet_, hyperG_vldt_, numRRsets_);
        double degVldt = infVldt * numRRsets_ / graph_->num_nodes;

        double upperBound = infSelf / approx;
        if (mode == 1) upperBound = boundLast_;
        else if (mode == 2) upperBound = boundMin_;

        double upperDegOPT = upperBound * numRRsets_ / graph_->num_nodes;

        double lowerSelect = std::pow(std::sqrt(degVldt + a1 * 2.0 / 9.0) - std::sqrt(a1 / 2.0), 2) - a1 / 18.0;
        double upperOPT = std::pow(std::sqrt(upperDegOPT + a2 / 2.0) + std::sqrt(a2 / 2.0), 2);

        approximation_ = lowerSelect / upperOPT;
        influence_ = infVldt;

        if (verbose_) {
            std::cerr << "[OPIM] Approximation guarantee: " << approximation_ << std::endl;
            std::cerr << "[OPIM] Influence (via R2): " << influence_ << std::endl;
            std::cerr << "[OPIM] Total RR sets: " << numRRsets << std::endl;
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

    std::set<int> run(int k, double epsilon, double delta = -1.0, int mode = 2) {
        if (k <= 0) return {};

        if (delta < 0) {
            delta = 1.0 / graph_->num_nodes;
        }

        const double e = std::exp(1.0);
        const double approx = 1.0 - 1.0 / e;

        double alpha = std::sqrt(std::log(6.0 / delta));
        double beta = std::sqrt((1.0 - 1.0/e) * (logCnk(graph_->num_nodes, k) + std::log(6.0 / delta)));

        int64_t numRbase = static_cast<int64_t>(2.0 * std::pow((1.0 - 1.0/e) * alpha + beta, 2));
        int64_t maxNumR = static_cast<int64_t>(2.0 * graph_->num_nodes * 
            std::pow((1.0 - 1.0/e) * alpha + beta, 2) / k / (epsilon * epsilon)) + 1;

        int numIter = static_cast<int>(std::log2(static_cast<double>(maxNumR) / numRbase)) + 1;

        const double a1 = std::log(numIter * 3.0 / delta);
        const double a2 = std::log(numIter * 3.0 / delta);

        if (verbose_) {
            std::cerr << "[OPIM-C] numRbase: " << numRbase << ", maxNumR: " << maxNumR 
                      << ", numIter: " << numIter << std::endl;
        }

        for (int idx = 0; idx < numIter; idx++) {
            int64_t numR = numRbase << idx;

            if (verbose_) {
                std::cerr << "[OPIM-C] Iteration " << (idx + 1) << "/" << numIter 
                          << ": sampling " << numR << " RR sets..." << std::endl;
            }

            hyperGT_.clear();
            hyperG_.clear();
            hyperGT_vldt_.clear();
            hyperG_vldt_.clear();

            hyperG_.resize(graph_->num_nodes);
            buildHyperGraphR(numR);
            buildHyperGraphRVldt(numR);

            numRRsets_ = static_cast<int64_t>(hyperGT_.size());

            double infSelf = maxCoverLazy(k, mode, hyperGT_, hyperG_, numRRsets_);

            double infVldt = selfInfCal(seedSet_, hyperG_vldt_, numRRsets_);
            double degVldt = infVldt * numRRsets_ / graph_->num_nodes;

            double upperBound = infSelf / approx;
            if (mode == 1) upperBound = boundLast_;
            else if (mode == 2) upperBound = boundMin_;

            double upperDegOPT = upperBound * numRRsets_ / graph_->num_nodes;

            double lowerSelect = std::pow(std::sqrt(degVldt + a1 * 2.0 / 9.0) - std::sqrt(a1 / 2.0), 2) - a1 / 18.0;
            double upperOPT = std::pow(std::sqrt(upperDegOPT + a2 / 2.0) + std::sqrt(a2 / 2.0), 2);

            double approxOPIMC = lowerSelect / upperOPT;

            if (verbose_) {
                std::cerr << "[OPIM-C]   approx: " << approxOPIMC 
                          << " (target: " << (approx - epsilon) << ")" << std::endl;
            }

            if (approxOPIMC >= approx - epsilon) {
                approximation_ = approxOPIMC;
                influence_ = infVldt;
                if (verbose_) {
                    std::cerr << "[OPIM-C] Converged! Approximation: " << approximation_ 
                              << ", Influence: " << influence_ << std::endl;
                }
                return seedSet_;
            }
        }

        return seedSet_;
    }
};

} // namespace pynetim
