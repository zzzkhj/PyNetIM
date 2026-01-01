// bindings/lt_binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "diffusion_model.h"  // м╛ио

namespace py = pybind11;

PYBIND11_MODULE(linear_threshold_model, m) {
    m.doc() = "Linear Threshold (LT) Influence Maximization Model";

    py::class_<LinearThresholdModel>(m, "LinearThresholdModel")
        .def(py::init<const Graph&, const std::set<int>&, double, double>(),
             py::arg("graph"), py::arg("seeds"),
             py::arg("theta_l") = 0.0, py::arg("theta_h") = 1.0,
             R"doc(
             Construct LT model.
             Each node gets a random threshold uniformly from [theta_l, theta_h).
             )doc")

        .def("set_seeds", &LinearThresholdModel::set_seeds,
             py::arg("new_seeds"),
             "Update the seed set")

        .def("run_monte_carlo_diffusion",
             &LinearThresholdModel::run_monte_carlo_diffusion,
             py::arg("rounds"),
             py::arg("seed") = 0,
             py::arg("use_multithread") = false,
             R"doc(
             Run Monte Carlo simulation of LT diffusion.
             Returns average number of activated nodes.
             )doc");
}
