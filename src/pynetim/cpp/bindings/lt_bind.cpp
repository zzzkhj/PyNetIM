#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "diffusion_model.h"

namespace py = pybind11;

PYBIND11_MODULE(linear_threshold_model, m) {
    m.doc() = "Linear Threshold (LT) Influence Maximization Model";

    py::class_<pynetim::LinearThresholdModel>(m, "LinearThresholdModel")
        .def(py::init<std::shared_ptr<pynetim::Graph>, const std::set<int>&, double, double>(),
             py::arg("graph"), py::arg("seeds"),
             py::arg("theta_l") = 0.0, py::arg("theta_h") = 1.0,
             py::keep_alive<0, 1>(),
             R"doc(
             Construct LT model.
             Each node gets a random threshold uniformly from [theta_l, theta_h).
             )doc")

        .def("set_seeds", &pynetim::LinearThresholdModel::set_seeds,
             py::arg("new_seeds"),
             "Update seed set")

        .def("run_monte_carlo_diffusion",
            &pynetim::LinearThresholdModel::run_monte_carlo_diffusion,
            py::arg("rounds"),
            py::arg("seed") = 0,
            py::arg("use_multithread") = false,
            py::arg("num_threads") = 0,
            R"doc(
             Run Monte Carlo simulation of LT diffusion.
             Returns average number of activated nodes.
             num_threads: number of threads to use (0 = auto-detect)
             )doc");
}