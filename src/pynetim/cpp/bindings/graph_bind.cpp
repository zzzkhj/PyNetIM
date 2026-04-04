#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "Graph.h"

namespace py = pybind11;

PYBIND11_MODULE(graph, m) {
    m.doc() = "C++ Graph implementation exposed to Python via pybind11";

    py::class_<pynetim::Edge>(m, "Edge")
        .def_readonly("to", &pynetim::Edge::to, "Target node")
        .def_readonly("weight", &pynetim::Edge::weight, "Edge weight")
        .def("__repr__", [](const pynetim::Edge& e) {
            return py::str("Edge(to={}, weight={})").format(e.to, e.weight);
        });

    py::class_<pynetim::Graph, std::shared_ptr<pynetim::Graph>>(m, "IMGraphCpp")
        .def(py::init([](int num_nodes,
                         py::object edges_obj,
                         py::object weights_obj,
                         bool directed) {
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
            }
            
            if (weights_obj.is_none()) {
                return std::make_shared<pynetim::Graph>(num_nodes, edges, std::vector<double>{}, directed);
            } else if (py::isinstance<py::float_>(weights_obj) || py::isinstance<py::int_>(weights_obj)) {
                double uniform_weight = weights_obj.cast<double>();
                return std::make_shared<pynetim::Graph>(num_nodes, edges, uniform_weight, directed);
            } else {
                std::vector<double> weights = weights_obj.cast<std::vector<double>>();
                return std::make_shared<pynetim::Graph>(num_nodes, edges, weights, directed);
            }
        }),
            py::arg("num_nodes"),
            py::arg("edges"),
            py::arg("weights") = 1.0,
            py::arg("directed") = true,
            R"doc(
            Construct a graph.
            
            Parameters
            ----------
            num_nodes : int
                Number of nodes in the graph
            edges : list of (int, int)
                List of edges as (u, v) tuples
            weights : list of float or float, optional
                If list: weights for each edge
                If float: uniform weight for all edges (default: 1.0)
            directed : bool, optional
                Whether the graph is directed (default: True)
            
            Examples
            --------
            >>> g = IMGraphCpp(3, [(0, 1), (1, 2)])  # all weights = 1.0
            >>> g = IMGraphCpp(3, [(0, 1), (1, 2)], [0.5, 0.3])  # individual weights
            >>> g = IMGraphCpp(3, [(0, 1), (1, 2)], 0.5)  # uniform weight 0.5
            )doc"
        )

        .def_readonly("num_nodes", &pynetim::Graph::num_nodes, "Number of nodes")
        .def_readonly("num_edges", &pynetim::Graph::num_edges, "Number of edges")
        .def_readonly("directed", &pynetim::Graph::directed, "Whether the graph is directed")
        .def_readonly("edges", &pynetim::Graph::edges, "Edges and weights of graph")

        .def("add_edge", &pynetim::Graph::add_edge,
            py::arg("u"), py::arg("v"), py::arg("w") = 1.0,
            "Add a weighted edge u -> v (or u - v if undirected)")

        .def("add_edges",
            &pynetim::Graph::add_edges,
            py::arg("edges"), py::arg("weights") = std::vector<double>{},
            "Add multiple edges at once. edges is a list of (u, v) tuples.")

        .def("update_edge_weight", &pynetim::Graph::update_edge_weight,
            py::arg("u"), py::arg("v"), py::arg("w"),
            "Update the weight of an existing edge")

        .def("remove_edge",
            &pynetim::Graph::remove_edge,
            py::arg("u"),
            py::arg("v"))

        .def("remove_edges",
            &pynetim::Graph::remove_edges,
            py::arg("edges"))

        .def("out_neighbors", &pynetim::Graph::out_neighbors,
            py::return_value_policy::reference_internal,
            py::arg("u"),
            "Return outgoing neighbors of node u (as list of Edge objects)")

        .def("in_neighbors", &pynetim::Graph::in_neighbors,
            py::return_value_policy::reference_internal,
            py::arg("u"),
            "Return incoming neighbors of node u (as list of node IDs)")

        .def("out_degree", &pynetim::Graph::out_degree, py::arg("u"))
        .def("in_degree", &pynetim::Graph::in_degree, py::arg("u"))
        .def("degree", &pynetim::Graph::degree, py::arg("u"))

        .def("get_all_degrees", &pynetim::Graph::get_all_degrees,
            "Return degree of all nodes as a list")
        .def("get_all_in_degrees", &pynetim::Graph::get_all_in_degrees,
            "Return in-degree of all nodes as a list")
        .def("get_all_out_degrees", &pynetim::Graph::get_all_out_degrees,
            "Return out-degree of all nodes as a list")

        .def("batch_out_degree", &pynetim::Graph::batch_out_degree,
            py::arg("nodes"),
            "Return out-degrees of specified nodes")

        .def("batch_in_degree", &pynetim::Graph::batch_in_degree,
            py::arg("nodes"),
            "Return in-degrees of specified nodes")

        .def("batch_degree", &pynetim::Graph::batch_degree,
            py::arg("nodes"),
            "Return degrees of specified nodes")

        .def("batch_out_neighbors", &pynetim::Graph::batch_out_neighbors,
            py::arg("nodes"),
            "Return outgoing neighbors of specified nodes (list of lists of (neighbor, weight) tuples)")

        .def("batch_get_edge_weight", &pynetim::Graph::batch_get_edge_weight,
            py::arg("edges"),
            "Return weights of specified edges (list of (u, v) tuples)")

        .def("get_adj_list", &pynetim::Graph::get_adj_list,
            py::return_value_policy::reference_internal,
            "Return the full adjacency list (list of lists of Edge objects)")

        .def("get_adj_matrix", &pynetim::Graph::get_adj_matrix,
            "Return dense adjacency matrix (list of lists of float)")

        .def("get_adj_matrix_sparse", &pynetim::Graph::get_adj_matrix_sparse,
            "Return sparse adjacency matrix (list of (u, v, weight) tuples)")

        .def("get_edge_weight", &pynetim::Graph::get_edge_weight,
            py::arg("u"), py::arg("v"),
            "Get the weight of edge (u, v)")

        .def("__repr__", &pynetim::Graph::__repr__);
}