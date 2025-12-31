// bindings/ic_binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "diffusion_model.h"  // 或者 #include "diffusion_models.h" 如果你分离了头文件
// 假设 Graph.h 已经在 diffusion_models.cpp 中被包含

namespace py = pybind11;

PYBIND11_MODULE(independent_cascade_model, m) {
    m.doc() = "Independent Cascade (IC) Influence Maximization Model";

    py::class_<IndependentCascadeModel>(m, "IndependentCascadeModel")
        .def(py::init<const std::set<int>&, const Graph&>(),
            py::arg("seeds"), py::arg("graph"),
            "Construct IC model with initial seed set and graph")

        .def("set_seeds", &IndependentCascadeModel::set_seeds,
            py::arg("new_seeds"),
            "Update the seed set")

        .def("run_monte_carlo_diffusion",
            &IndependentCascadeModel::run_monte_carlo_diffusion,
            py::arg("rounds"),
            py::arg("seed") = 0,
            py::arg("use_multithread") = false,
            R"doc(
             Run Monte Carlo simulation of IC diffusion.
             Returns average number of activated nodes over 'rounds' trials.
             Results are deterministic for the same seed (single/multi-thread).
             )doc");
}
