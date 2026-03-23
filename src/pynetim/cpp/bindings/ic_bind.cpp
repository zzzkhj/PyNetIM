#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "diffusion_model.h"

namespace py = pybind11;

PYBIND11_MODULE(independent_cascade_model, m) {
    m.doc() = "Independent Cascade (IC) Influence Maximization Model";

    py::class_<pynetim::IndependentCascadeModel>(m, "IndependentCascadeModel")
        .def(py::init<const pynetim::Graph&, const std::set<int>&>(),
            py::arg("graph"), py::arg("seeds"),
            "Construct IC model with initial seed set and graph")

        .def("set_seeds", &pynetim::IndependentCascadeModel::set_seeds,
            py::arg("new_seeds"),
            "Update seed set")

        .def("run_monte_carlo_diffusion",
            &pynetim::IndependentCascadeModel::run_monte_carlo_diffusion,
            py::arg("rounds"),
            py::arg("seed") = 0,
            py::arg("use_multithread") = false,
            py::arg("num_threads") = 0,
            R"doc(
             Run Monte Carlo simulation of IC diffusion.
             Returns average number of activated nodes over 'rounds' trials.
             Results are deterministic for the same seed (single/multi-thread).
             num_threads: number of threads to use (0 = auto-detect)
             )doc");
}