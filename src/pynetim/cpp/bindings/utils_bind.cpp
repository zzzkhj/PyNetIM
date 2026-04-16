#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "utils.h"

namespace py = pybind11;

PYBIND11_MODULE(utils, m) {
    m.doc() = "PyNetIM 工具函数模块";

    {
        py::options options;
        options.disable_function_signatures();

        m.def("renumber_edges", [](py::object edges_obj, bool return_mapping = false) -> py::object {
            std::vector<std::tuple<int, int>> edges_list = edges_obj.cast<std::vector<std::tuple<int, int>>>();
            
            if (return_mapping) {
                auto [edges, o2i, i2o] = pynetim::utils::renumber_edges_full(edges_list);
                return py::make_tuple(edges, o2i, i2o);
            } else {
                return py::cast(pynetim::utils::renumber_edges_only(edges_list));
            }
        }, py::arg("edges"), py::arg("return_mapping") = false,
            R"doc(renumber_edges(edges: list[tuple[int, int]], return_mapping: bool = False) -> list[tuple[int, int]] | tuple[list[tuple[int, int]], dict, dict]

将边列表重新编号为从 0 开始的连续节点 ID。

Args:
    edges: 边列表，每个元素为 (u, v) 元组，节点 ID 可以是任意整数。
    return_mapping: 是否返回映射关系，默认为 False。
        若为 False，仅返回重新编号后的边列表。
        若为 True，返回 (边列表, 原始到内部映射, 内部到原始映射)。

Returns:
    若 return_mapping=False：返回重新编号后的边列表。
    若 return_mapping=True：返回 (edges, original_to_internal, internal_to_original)。

Example:
    >>> from pynetim.utils import renumber_edges
    >>> edges = [(1, 2), (3, 4), (2, 5)]
    >>> new_edges = renumber_edges(edges)
    >>> new_edges, o2i, i2o = renumber_edges(edges, return_mapping=True)
)doc");
    }
}
