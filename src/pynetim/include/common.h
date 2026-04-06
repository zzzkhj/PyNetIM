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
#include <array>
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

inline std::mt19937 create_rng(bool use_random_seed, unsigned int seed) {
    std::mt19937 rng;
    if (use_random_seed) {
        std::random_device rd;
        std::array<unsigned int, 4> seed_data;
        for (auto& item : seed_data) {
            item = rd();
        }
        std::seed_seq seq(seed_data.begin(), seed_data.end());
        rng.seed(seq);
    } else {
        rng.seed(seed);
    }
    return rng;
}

inline std::vector<unsigned int> generate_trial_seeds(int rounds, unsigned int seed) {
    std::vector<unsigned int> trial_seeds(rounds);
    std::mt19937 master_rng;
    if (seed == 0) {
        std::random_device rd;
        std::array<unsigned int, 4> seed_data;
        for (auto& item : seed_data) {
            item = rd();
        }
        std::seed_seq seq(seed_data.begin(), seed_data.end());
        master_rng.seed(seq);
    } else {
        master_rng.seed(seed);
    }
    for (int i = 0; i < rounds; ++i) {
        trial_seeds[i] = master_rng();
    }
    return trial_seeds;
}

}
