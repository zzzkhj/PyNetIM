#pragma once
#include <vector>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <variant>

namespace pynetim {
namespace utils {

inline std::vector<std::tuple<int, int>> renumber_edges_only(
    const std::vector<std::tuple<int, int>>& edges_list) {
    
    std::unordered_set<int> unique_nodes;
    for (const auto& e : edges_list) {
        unique_nodes.insert(std::get<0>(e));
        unique_nodes.insert(std::get<1>(e));
    }
    
    std::vector<int> sorted_nodes(unique_nodes.begin(), unique_nodes.end());
    std::sort(sorted_nodes.begin(), sorted_nodes.end());
    
    std::unordered_map<int, int> mapping;
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
        mapping[sorted_nodes[i]] = static_cast<int>(i);
    }
    
    std::vector<std::tuple<int, int>> result;
    result.reserve(edges_list.size());
    for (const auto& e : edges_list) {
        result.emplace_back(mapping[std::get<0>(e)], mapping[std::get<1>(e)]);
    }
    
    return result;
}

inline std::tuple<
    std::vector<std::tuple<int, int>>,
    std::unordered_map<int, int>,
    std::vector<int>
> renumber_edges_full(
    const std::vector<std::tuple<int, int>>& edges_list) {
    
    std::unordered_set<int> unique_nodes;
    for (const auto& e : edges_list) {
        unique_nodes.insert(std::get<0>(e));
        unique_nodes.insert(std::get<1>(e));
    }
    
    std::vector<int> sorted_nodes(unique_nodes.begin(), unique_nodes.end());
    std::sort(sorted_nodes.begin(), sorted_nodes.end());
    
    std::unordered_map<int, int> original_to_internal;
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
        original_to_internal[sorted_nodes[i]] = static_cast<int>(i);
    }
    
    std::vector<std::tuple<int, int>> edges;
    edges.reserve(edges_list.size());
    for (const auto& e : edges_list) {
        edges.emplace_back(
            original_to_internal[std::get<0>(e)],
            original_to_internal[std::get<1>(e)]
        );
    }
    
    return {edges, original_to_internal, sorted_nodes};
}

}
}
