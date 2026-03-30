#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "diffusion_model.h"

namespace py = pybind11;

PYBIND11_MODULE(independent_cascade_model, m) {
    m.doc() = "Independent Cascade (IC) Influence Maximization Model";

    py::class_<pynetim::IndependentCascadeModel>(m, "IndependentCascadeModel")
        .def(py::init([](py::object graph_obj, const std::set<int>& seeds, bool record_activated) {
            auto graph_ptr = py::cast<std::shared_ptr<pynetim::Graph>>(graph_obj);
            return std::make_unique<pynetim::IndependentCascadeModel>(graph_ptr, seeds, graph_obj, record_activated);
        }),
            py::arg("graph"), py::arg("seeds"), py::arg("record_activated") = false,
            py::keep_alive<0, 1>(),
            "Construct IC model with initial seed set and graph")

        .def("set_seeds", &pynetim::IndependentCascadeModel::set_seeds,
            py::arg("new_seeds"),
            "Update seed set")

        .def("set_record_activated", &pynetim::IndependentCascadeModel::set_record_activated,
            py::arg("record"),
            "Enable or disable recording of activated nodes")

        .def("run_single_simulation",
            &pynetim::IndependentCascadeModel::run_single_simulation,
            py::arg("seed") = 0,
            "Run a single diffusion simulation and return set of activated nodes")

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