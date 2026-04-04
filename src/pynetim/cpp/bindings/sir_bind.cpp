#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "diffusion_model.h"

namespace py = pybind11;

PYBIND11_MODULE(susceptible_infected_recovered_model, m) {
    m.doc() = "Susceptible-Infected-Recovered (SIR) Epidemic Model";

    py::class_<pynetim::SusceptibleInfectedRecoveredModel>(m, "SusceptibleInfectedRecoveredModel")
        .def(py::init([](py::object graph_obj, const std::set<int>& seeds, double beta, double gamma, bool record_activated, bool record_activation_frequency) {
            auto graph_ptr = py::cast<std::shared_ptr<pynetim::Graph>>(graph_obj);
            return std::make_unique<pynetim::SusceptibleInfectedRecoveredModel>(graph_ptr, seeds, beta, gamma, record_activated, record_activation_frequency);
        }),
            py::arg("graph"), py::arg("seeds"), py::arg("beta") = 1.0, py::arg("gamma") = 0.0, py::arg("record_activated") = false, py::arg("record_activation_frequency") = false,
            py::keep_alive<0, 1>(),
            "Construct SIR model with initial seed set and graph")

        .def("set_seeds", &pynetim::SusceptibleInfectedRecoveredModel::set_seeds,
            py::arg("new_seeds"),
            "Update seed set")

        .def("set_beta", &pynetim::SusceptibleInfectedRecoveredModel::set_beta,
            py::arg("beta"),
            "Set infection probability")

        .def("set_gamma", &pynetim::SusceptibleInfectedRecoveredModel::set_gamma,
            py::arg("gamma"),
            "Set recovery probability")

        .def("set_record_activated", &pynetim::SusceptibleInfectedRecoveredModel::set_record_activated,
            py::arg("record"),
            "Enable or disable recording of activated nodes")

        .def("set_record_activation_frequency", &pynetim::SusceptibleInfectedRecoveredModel::set_record_activation_frequency,
            py::arg("record"),
            "Enable or disable recording of activation frequency")

        .def("run_single_simulation",
            [](const pynetim::SusceptibleInfectedRecoveredModel& self, py::object seed_obj) {
                bool use_random_seed = seed_obj.is_none();
                unsigned int seed = use_random_seed ? 0 : py::cast<unsigned int>(seed_obj);
                return self.run_single_simulation(use_random_seed, seed);
            },
            py::arg("seed") = py::none(),
            R"doc(
             Run a single diffusion simulation and return number of infected + recovered nodes.
             
             Parameters
             ----------
             seed : int or None, optional
                 Random seed for reproducibility. If None (default), uses a truly random seed.
                 Use the same seed to get reproducible results.
             
             Returns
             -------
             int
                 Number of infected + recovered nodes in this simulation
             
             Examples
             --------
             >>> count = model.run_single_simulation()  # Random result
             >>> count = model.run_single_simulation(seed=42)  # Reproducible result
             >>> activated_nodes = model.get_activated_nodes()  # Get recorded nodes (if record_activated=True)
             )doc")

        .def("get_activated_nodes",
            &pynetim::SusceptibleInfectedRecoveredModel::get_activated_nodes,
            R"doc(
             Get set of infected + recovered nodes from last simulation.
             
             Returns
             -------
             Set[int]
                 Set of infected + recovered nodes from the last simulation.
                 For single simulation: nodes infected or recovered in that simulation.
                 For Monte Carlo: union of all infected/recovered nodes across all trials.
                 Only valid when record_activated is set to True.
             
             Examples
             --------
             >>> model.set_record_activated(True)
             >>> count = model.run_single_simulation()
             >>> activated_nodes = model.get_activated_nodes()
             >>> print(f"Activated {count} nodes: {activated_nodes}")
             )doc")

        .def("get_activation_frequency",
            &pynetim::SusceptibleInfectedRecoveredModel::get_activation_frequency,
            R"doc(
             Get activation frequency of each node from all simulations.
             
             Returns
             -------
             List[int]
                 List where index i represents how many times node i was infected or recovered.
                 Only valid when record_activation_frequency is set to True.
             
             Examples
             --------
             >>> model.set_record_activation_frequency(True)
             >>> avg = model.run_monte_carlo_diffusion(1000)
             >>> freq = model.get_activation_frequency()
             >>> print(f"Node 0 was activated {freq[0]} times")
             )doc")

        .def("run_monte_carlo_diffusion",
            [](const pynetim::SusceptibleInfectedRecoveredModel& self, int rounds, py::object seed_obj, bool use_multithread = false, int num_threads = 0) {
                bool use_random_seed = seed_obj.is_none();
                unsigned int seed = use_random_seed ? 0 : py::cast<unsigned int>(seed_obj);
                return self.run_monte_carlo_diffusion(rounds, seed, use_multithread, num_threads);
            },
            py::arg("rounds"),
            py::arg("seed") = py::none(),
            py::arg("use_multithread") = false,
            py::arg("num_threads") = 0,
            R"doc(
             Run Monte Carlo simulation of SIR diffusion.
             Returns average number of infected + recovered nodes over 'rounds' trials.
             
             Parameters
             ----------
             rounds : int
                 Number of simulation trials
             seed : int or None, optional
                 Random seed for reproducibility. If None (default), uses a truly random seed.
                 Use the same seed to get reproducible results.
             use_multithread : bool, optional
                 Whether to use multithreading (default: False)
             num_threads : int, optional
                 Number of threads to use (0 = auto-detect, default: 0)
             
             Returns
             -------
             float
                 Average number of infected + recovered nodes over all trials
             
             Examples
             --------
             >>> avg = model.run_monte_carlo_diffusion(1000)  # Random seed
             >>> avg = model.run_monte_carlo_diffusion(1000, seed=42)  # Reproducible result
             >>> avg = model.run_monte_carlo_diffusion(1000, seed=42, use_multithread=True, num_threads=4)
             )doc");
}
