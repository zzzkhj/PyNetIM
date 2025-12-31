// graph_binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>          // 自动转换 std::vector, std::unordered_map 等
#include <pybind11/operators.h>    // 支持运算符重载（如 ==）

#include "Graph.h"     // 假设你的 Graph 类定义在 Graph.h 中
// 如果上面的代码就是完整的头文件，直接把上面的代码保存为 Graph.h

namespace py = pybind11;

PYBIND11_MODULE(graph, m) {
    m.doc() = "C++ Graph implementation exposed to Python via pybind11";

    py::class_<Graph>(m, "IMGraphCpp")
        /* ===== 构造函数 ===== */
        .def(py::init<
            int,
            const std::vector<std::tuple<int, int>>&,
            const std::vector<double>&,
            bool
        >(),
            py::arg("num_nodes"),
            py::arg("edges"),
            py::arg("weights") = std::vector<double>{},
            py::arg("directed") = true
        )


        // 只读属性
        .def_readonly("num_nodes", &Graph::num_nodes, "Number of nodes")
        .def_readonly("num_edges", &Graph::num_edges, "Number of edges")
        .def_readonly("directed", &Graph::directed, "Whether the graph is directed")
        .def_readonly("edges", &Graph::edges, "Edges and weights of graph")

        // 构建图
        .def("add_edge", &Graph::add_edge,
            py::arg("u"), py::arg("v"), py::arg("w") = 1.0,
            "Add a weighted edge u -> v (or u - v if undirected)")

        .def("add_edges",
            py::overload_cast<
            const std::vector<std::tuple<int, int>>&,
            const std::vector<double>&>(&Graph::add_edges),
            py::arg("edges"), py::arg("weights") = std::vector<double>{},
            "Add multiple edges at once. edges is a list of (u, v) tuples.")

        .def("update_edge_weight", &Graph::update_edge_weight,
            py::arg("u"), py::arg("v"), py::arg("w"),
            "Update the weight of an existing edge")

        // 删除
        .def("remove_edge",
            &Graph::remove_edge,
            py::arg("u"),
            py::arg("v"))

        .def("remove_edges",
            &Graph::remove_edges,
            py::arg("edges"))

        // 查询接口
        .def("out_neighbors", &Graph::out_neighbors,
            py::return_value_policy::reference_internal,
            py::arg("u"),
            "Return outgoing neighbors of node u (as set)")

        .def("in_neighbors", &Graph::in_neighbors,
            py::return_value_policy::reference_internal,
            py::arg("u"),
            "Return incoming neighbors of node u (as set)")

        .def("out_degree", &Graph::out_degree, py::arg("u"))
        .def("in_degree", &Graph::in_degree, py::arg("u"))
        .def("degree", &Graph::degree, py::arg("u"))

        .def("get_adj_list", &Graph::get_adj_list,
            py::return_value_policy::reference_internal,
            "Return the full adjacency list (list of sets)")

        .def("get_adj_matrix", &Graph::get_adj_matrix,
            "Return dense adjacency matrix (list of lists of float)")

        // __repr__
        .def("__repr__", &Graph::__repr__);
}
