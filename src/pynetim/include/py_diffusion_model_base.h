#pragma once
#include "diffusion_model_base.h"

namespace pynetim {

class PyDiffusionModelBase : public DiffusionModelBase {
private:
    py::object py_override;
    py::object cpp_graph;

protected:
    int run_single_trial(
        std::mt19937& rng,
        std::uniform_real_distribution<double>& dist,
        const std::set<int>& trial_seeds,
        std::set<int>* activated_nodes = nullptr,
        std::vector<int>* activation_count = nullptr) const override {

        py::gil_scoped_acquire gil;

        py::list seeds_list;
        for (int s : trial_seeds) {
            seeds_list.append(s);
        }

        if (!py_override.is_none() && py::hasattr(py_override, "run_single_trial")) {
            py_override.attr("graph") = cpp_graph;

            py::object result = py_override.attr("run_single_trial")(
                seeds_list,
                py::cast(static_cast<unsigned int>(rng()))
            );

            auto tuple_result = py::cast<py::tuple>(result);
            int count = py::cast<int>(tuple_result[0]);

            if (activated_nodes != nullptr) {
                *activated_nodes = py::cast<std::set<int>>(tuple_result[1]);
            }

            if (activation_count != nullptr) {
                py::list count_list = py::cast<py::list>(tuple_result[2]);
                activation_count->clear();
                activation_count->reserve(py::len(count_list));
                for (auto item : count_list) {
                    activation_count->push_back(py::cast<int>(item));
                }
            }

            return count;
        }

        PyErr_SetString(PyExc_NotImplementedError, "run_single_trial must be implemented");
        throw py::error_already_set();
    }

public:
    PyDiffusionModelBase(
        std::shared_ptr<Graph> graph_ptr,
        const std::set<int>& seeds,
        py::object py_override = py::none(),
        bool record_activated = false,
        bool record_activation_frequency = false)
        : DiffusionModelBase(graph_ptr, seeds, record_activated, record_activation_frequency),
          py_override(py_override) {
        cpp_graph = py::cast(graph_ptr);
    }

    void set_py_override(py::object obj) {
        py_override = obj;
    }

    py::object get_py_override() const {
        return py_override;
    }

    py::object graph() const {
        return cpp_graph;
    }
};

}