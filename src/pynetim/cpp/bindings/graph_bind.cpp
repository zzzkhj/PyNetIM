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
                         const std::vector<std::tuple<int, int>>& edges,
                         const std::vector<double>& weights,
                         bool directed) {
            return std::make_shared<pynetim::Graph>(num_nodes, edges, weights, directed);
        }),
            py::arg("num_nodes"),
            py::arg("edges"),
            py::arg("weights") = std::vector<double>{},
            py::arg("directed") = true
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