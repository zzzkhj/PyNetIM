#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "graph/Graph.h"

namespace py = pybind11;

PYBIND11_MODULE(graph, m) {
    m.doc() = "图结构模块，提供高效的 C++ 图实现";

    py::class_<pynetim::Edge>(m, "Edge")
        .def_readonly("to", &pynetim::Edge::to,
            R"doc(目标节点。)doc")
        .def_readonly("weight", &pynetim::Edge::weight,
            R"doc(边权重。)doc")
        .def("__repr__", [](const pynetim::Edge& e) {
            return py::str("Edge(to={}, weight={})").format(e.to, e.weight);
        });

    {
        py::options options;
        options.disable_function_signatures();

        py::class_<pynetim::Graph, std::shared_ptr<pynetim::Graph>>(m, "IMGraph")
            .def(py::init([](py::object edges_obj,
                             py::object weights_obj,
                             bool directed,
                             bool renumber) {
                if (!py::isinstance<py::list>(edges_obj) && !py::isinstance<py::tuple>(edges_obj)) {
                    throw py::type_error("IMGraph() 参数错误: edges 必须是 list[tuple[int, int]] 类型。\n用法: IMGraph(edges, weights=1.0, directed=True, renumber=True)");
                }
                
                std::vector<std::tuple<int, int>> edges;
                if (py::isinstance<py::list>(edges_obj)) {
                    for (auto item : edges_obj.cast<py::list>()) {
                        if (py::isinstance<py::tuple>(item)) {
                            auto t = item.cast<py::tuple>();
                            edges.emplace_back(t[0].cast<int>(), t[1].cast<int>());
                        } else if (py::isinstance<py::list>(item)) {
                            auto l = item.cast<py::list>();
                            edges.emplace_back(l[0].cast<int>(), l[1].cast<int>());
                        }
                    }
                } else if (py::isinstance<py::tuple>(edges_obj)) {
                    for (auto item : edges_obj.cast<py::tuple>()) {
                        if (py::isinstance<py::tuple>(item)) {
                            auto t = item.cast<py::tuple>();
                            edges.emplace_back(t[0].cast<int>(), t[1].cast<int>());
                        } else if (py::isinstance<py::list>(item)) {
                            auto l = item.cast<py::list>();
                            edges.emplace_back(l[0].cast<int>(), l[1].cast<int>());
                        }
                    }
                }
                
                if (weights_obj.is_none()) {
                    return std::make_shared<pynetim::Graph>(edges, std::vector<double>{}, directed, renumber);
                } else if (py::isinstance<py::float_>(weights_obj) || py::isinstance<py::int_>(weights_obj)) {
                    double uniform_weight = weights_obj.cast<double>();
                    return std::make_shared<pynetim::Graph>(edges, uniform_weight, directed, renumber);
                } else if (py::isinstance<py::list>(weights_obj)) {
                    std::vector<double> weights = weights_obj.cast<std::vector<double>>();
                    return std::make_shared<pynetim::Graph>(edges, weights, directed, renumber);
                } else {
                    throw py::type_error("IMGraph() 参数错误: weights 必须是 float 或 list[float] 类型。\n用法: IMGraph(edges, weights=1.0, directed=True, renumber=True)");
                }
            }),
                py::arg("edges"),
                py::arg("weights") = 1.0,
                py::arg("directed") = true,
                py::arg("renumber") = true,
                R"doc(__init__(edges: list[tuple[int, int]], weights: float | list[float] = 1.0, directed: bool = True, renumber: bool = True) -> None

从边列表构建图。

传入的边列表必须是从 0 开始连续编号节点，若不是请设置 renumber=True。
或者在构建图之前使用 renumber_edges 函数重新编号节点。

Args:
    edges: 边列表，每个元素为 (u, v) 元组。
    weights: 边权重。若为列表则指定每条边的权重；若为浮点数则统一权重。默认为 1.0。
    directed: 是否为有向图，默认为 True。
    renumber: 是否重新编号节点。若为 True，将节点重新编号为从 0 开始的连续整数。默认为 True。

Returns:
    IMGraph: 图对象。
)doc"
            )

            .def_readonly("num_nodes", &pynetim::Graph::num_nodes,
                R"doc(节点数量。)doc")
            .def_readonly("num_edges", &pynetim::Graph::num_edges,
                R"doc(边数量。)doc")
            .def_readonly("directed", &pynetim::Graph::directed,
                R"doc(是否为有向图。)doc")
            .def_readonly("edges", &pynetim::Graph::edges,
                R"doc(边列表及权重。)doc")
            .def_readonly("original_to_internal", &pynetim::Graph::original_to_internal,
                R"doc(原始节点 ID 到内部 ID 的映射。)doc")
            .def_readonly("internal_to_original", &pynetim::Graph::internal_to_original,
                R"doc(内部节点 ID 到原始 ID 的映射。)doc")

            .def("add_edge", &pynetim::Graph::add_edge,
                py::arg("u"), py::arg("v"), py::arg("w") = 1.0,
                R"doc(add_edge(u: int, v: int, w: float = 1.0) -> None

添加带权边。

Args:
    u: 源节点。
    v: 目标节点。
    w: 边权重，默认为 1.0。
)doc")

            .def("add_edges",
                &pynetim::Graph::add_edges,
                py::arg("edges"), py::arg("weights") = std::vector<double>{},
                R"doc(add_edges(edges: list[tuple[int, int]], weights: list[float] = []) -> None

批量添加边。

Args:
    edges: 边列表，每个元素为 (u, v) 元组。
    weights: 边权重列表。
)doc")

            .def("update_edge_weight", &pynetim::Graph::update_edge_weight,
                py::arg("u"), py::arg("v"), py::arg("w"),
                R"doc(update_edge_weight(u: int, v: int, w: float) -> None

更新已有边的权重。

Args:
    u: 源节点。
    v: 目标节点。
    w: 新的边权重。
)doc")

            .def("remove_edge",
                &pynetim::Graph::remove_edge,
                py::arg("u"),
                py::arg("v"),
                R"doc(remove_edge(u: int, v: int) -> None

删除边。

Args:
    u: 源节点。
    v: 目标节点。
)doc")

            .def("remove_edges",
                &pynetim::Graph::remove_edges,
                py::arg("edges"),
                R"doc(remove_edges(edges: list[tuple[int, int]]) -> None

批量删除边。

Args:
    edges: 边列表，每个元素为 (u, v) 元组。
)doc")

            .def("out_neighbors", &pynetim::Graph::out_neighbors_list,
                py::arg("u"),
                R"doc(out_neighbors(u: int) -> list[int]

返回节点的出边邻居。

Args:
    u: 节点 ID。

Returns:
    List[int]: 出边邻居节点 ID 列表。
)doc")

            .def("out_neighbors_with_weights", &pynetim::Graph::out_neighbors_fast,
                py::arg("u"),
                R"doc(out_neighbors_with_weights(u: int) -> list[tuple[int, float]]

返回节点的出边邻居及权重。

Args:
    u: 节点 ID。

Returns:
    List[Tuple[int, float]]: (邻居, 权重) 元组列表。
)doc")

            .def("out_neighbors_arrays", &pynetim::Graph::out_neighbors_arrays,
                py::arg("u"),
                R"doc(out_neighbors_arrays(u: int) -> tuple[list[int], list[float]]

返回节点的出边邻居（数组格式）。

Args:
    u: 节点 ID。

Returns:
    Tuple[List[int], List[float]]: 两个并行数组 (targets, weights)，O(1) 访问每个元素。
)doc")

            .def("in_neighbors", &pynetim::Graph::in_neighbors,
                py::return_value_policy::reference_internal,
                py::arg("u"),
                R"doc(in_neighbors(u: int) -> list[int]

返回节点的入边邻居。

Args:
    u: 节点 ID。

Returns:
    List[int]: 入边邻居节点 ID 列表。
)doc")

            .def("out_degree", &pynetim::Graph::out_degree,
                py::arg("u"),
                R"doc(out_degree(u: int) -> int

返回节点的出度。

Args:
    u: 节点 ID。

Returns:
    int: 出度。
)doc")

            .def("in_degree", &pynetim::Graph::in_degree,
                py::arg("u"),
                R"doc(in_degree(u: int) -> int

返回节点的入度。

Args:
    u: 节点 ID。

Returns:
    int: 入度。
)doc")

            .def("degree", &pynetim::Graph::degree,
                py::arg("u"),
                R"doc(degree(u: int) -> int

返回节点的度数。

Args:
    u: 节点 ID。

Returns:
    int: 度数。
)doc")

            .def("get_all_degrees", &pynetim::Graph::get_all_degrees,
                R"doc(get_all_degrees() -> list[int]

返回所有节点的度数列表。

Returns:
    List[int]: 度数列表。
)doc")

            .def("get_all_in_degrees", &pynetim::Graph::get_all_in_degrees,
                R"doc(get_all_in_degrees() -> list[int]

返回所有节点的入度列表。

Returns:
    List[int]: 入度列表。
)doc")

            .def("get_all_out_degrees", &pynetim::Graph::get_all_out_degrees,
                R"doc(get_all_out_degrees() -> list[int]

返回所有节点的出度列表。

Returns:
    List[int]: 出度列表。
)doc")

            .def("batch_out_degree", &pynetim::Graph::batch_out_degree,
                py::arg("nodes"),
                R"doc(batch_out_degree(nodes: list[int]) -> list[int]

批量返回指定节点的出度。

Args:
    nodes: 节点 ID 列表。

Returns:
    List[int]: 出度列表。
)doc")

            .def("batch_in_degree", &pynetim::Graph::batch_in_degree,
                py::arg("nodes"),
                R"doc(batch_in_degree(nodes: list[int]) -> list[int]

批量返回指定节点的入度。

Args:
    nodes: 节点 ID 列表。

Returns:
    List[int]: 入度列表。
)doc")

            .def("batch_degree", &pynetim::Graph::batch_degree,
                py::arg("nodes"),
                R"doc(batch_degree(nodes: list[int]) -> list[int]

批量返回指定节点的度数。

Args:
    nodes: 节点 ID 列表。

Returns:
    List[int]: 度数列表。
)doc")

            .def("batch_out_neighbors", &pynetim::Graph::batch_out_neighbors,
                py::arg("nodes"),
                R"doc(batch_out_neighbors(nodes: list[int]) -> list[list[int]]

批量返回指定节点的出边邻居。

Args:
    nodes: 节点 ID 列表。

Returns:
    List[List[int]]: 每个节点的出边邻居节点 ID 列表。
)doc")

            .def("batch_out_neighbors_with_weights", &pynetim::Graph::batch_out_neighbors_with_weights,
                py::arg("nodes"),
                R"doc(batch_out_neighbors_with_weights(nodes: list[int]) -> list[list[tuple[int, float]]]

批量返回指定节点的出边邻居及权重。

Args:
    nodes: 节点 ID 列表。

Returns:
    List[List[Tuple[int, float]]]: 每个节点的 (邻居, 权重) 元组列表。
)doc")

            .def("batch_get_edge_weight", &pynetim::Graph::batch_get_edge_weight,
                py::arg("edges"), py::arg("default_value") = 0.0, py::arg("raise_on_missing") = false,
                R"doc(batch_get_edge_weight(edges: list[tuple[int, int]], default_value: float = 0.0, raise_on_missing: bool = False) -> list[float]

批量返回指定边的权重。

Args:
    edges: 边列表，每个元素为 (u, v) 元组。
    default_value: 边不存在时的默认返回值，默认为 0.0。当 raise_on_missing=True 时忽略。
    raise_on_missing: 边不存在时是否抛出异常，默认为 False。

Returns:
    List[float]: 权重列表。

Raises:
    RuntimeError: 当 raise_on_missing=True 且边不存在时。
)doc")

            .def("get_adj_list", &pynetim::Graph::get_adj_list,
                py::return_value_policy::reference_internal,
                R"doc(get_adj_list() -> list[list[Edge]]

返回完整邻接表。

Returns:
    List[List[Edge]]: Edge 对象列表的列表。
)doc")

            .def("get_adj_list_py", &pynetim::Graph::get_adj_list_py,
                R"doc(get_adj_list_py() -> list[list[tuple[int, float]]]

返回 Python 友好格式的邻接表。

Returns:
    List[List[Tuple[int, float]]]: (邻居, 权重) 元组列表的列表。
)doc")

            .def("get_adj_matrix", &pynetim::Graph::get_adj_matrix,
                R"doc(get_adj_matrix() -> list[list[float]]

返回稠密邻接矩阵。

Returns:
    List[List[float]]: 邻接矩阵。
)doc")

            .def("get_adj_matrix_sparse", &pynetim::Graph::get_adj_matrix_sparse,
                R"doc(get_adj_matrix_sparse() -> list[tuple[int, int, float]]

返回稀疏邻接矩阵。

Returns:
    List[Tuple[int, int, float]]: (u, v, 权重) 元组列表。
)doc")

            .def("get_edge_weight", &pynetim::Graph::get_edge_weight,
                py::arg("u"), py::arg("v"),
                R"doc(get_edge_weight(u: int, v: int) -> float

获取边的权重。若边不存在则抛出异常。

Args:
    u: 源节点。
    v: 目标节点。

Returns:
    float: 边权重。

Raises:
    RuntimeError: 边不存在。
)doc")

            .def("has_edge", &pynetim::Graph::has_edge,
                py::arg("u"), py::arg("v"),
                R"doc(has_edge(u: int, v: int) -> bool

检查边是否存在。

Args:
    u: 源节点。
    v: 目标节点。

Returns:
    bool: 边存在返回 True，否则返回 False。
)doc")

            .def("__repr__", &pynetim::Graph::__repr__);
    }
}
