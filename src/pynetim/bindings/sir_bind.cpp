#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "diffusion_model.h"

namespace py = pybind11;

PYBIND11_MODULE(susceptible_infected_recovered_model, m) {
    m.doc() = "SIR 模型（易感-感染-恢复），用于流行病传播模拟";

    {
        py::options options;
        options.disable_function_signatures();

        py::class_<pynetim::SusceptibleInfectedRecoveredModel>(m, "SusceptibleInfectedRecoveredModel")
            .def(py::init([](py::object graph_obj, const std::set<int>& seeds, double beta, double gamma, bool record_activated, bool record_activation_frequency) {
                auto graph_ptr = py::cast<std::shared_ptr<pynetim::Graph>>(graph_obj);
                return std::make_unique<pynetim::SusceptibleInfectedRecoveredModel>(graph_ptr, seeds, beta, gamma, record_activated, record_activation_frequency);
            }),
                py::arg("graph"), py::arg("seeds"), py::arg("beta"), py::arg("gamma"), py::arg("record_activated") = false, py::arg("record_activation_frequency") = false,
                py::keep_alive<0, 1>(),
                R"doc(__init__(graph: IMGraph, seeds: set[int], beta: float, gamma: float, record_activated: bool = False, record_activation_frequency: bool = False) -> None

构造 SIR 模型。

Args:
    graph: 图对象。
    seeds: 初始感染节点集合。
    beta: 感染概率，必须在 (0, 1] 范围内。
    gamma: 恢复概率，必须在 (0, 1] 范围内。
    record_activated: 是否记录感染/恢复节点，默认为 False。
    record_activation_frequency: 是否记录感染/恢复频数，默认为 False。
)doc")

            .def("set_seeds", &pynetim::SusceptibleInfectedRecoveredModel::set_seeds,
                py::arg("new_seeds"),
                R"doc(set_seeds(new_seeds: set[int]) -> None

更新种子节点集合。

Args:
    new_seeds: 新的种子节点集合。
)doc")

            .def("set_record_activated", &pynetim::SusceptibleInfectedRecoveredModel::set_record_activated,
                py::arg("record"),
                R"doc(set_record_activated(record: bool) -> None

启用或禁用感染/恢复节点记录。

Args:
    record: 是否记录感染/恢复节点。
)doc")

            .def("set_record_activation_frequency", &pynetim::SusceptibleInfectedRecoveredModel::set_record_activation_frequency,
                py::arg("record"),
                R"doc(set_record_activation_frequency(record: bool) -> None

启用或禁用感染/恢复频数记录。

Args:
    record: 是否记录感染/恢复频数。
)doc")

            .def("run_single_simulation",
                [](pynetim::SusceptibleInfectedRecoveredModel& self, py::object seed_obj) {
                    bool use_random_seed = seed_obj.is_none();
                    unsigned int seed = use_random_seed ? 0 : py::cast<unsigned int>(seed_obj);
                    return self.run_single_simulation(use_random_seed, seed);
                },
                py::arg("seed") = py::none(),
                R"doc(run_single_simulation(seed: int | None = None) -> int

执行单次传播模拟。

Args:
    seed: 随机种子，用于结果可重现。若为 None 则使用真随机种子。

Returns:
    int: 本次模拟感染的节点数。
)doc")

            .def("get_activated_nodes",
                &pynetim::SusceptibleInfectedRecoveredModel::get_activated_nodes,
                R"doc(get_activated_nodes() -> set[int]

获取上次模拟的感染节点集合。

Returns:
    Set[int]: 感染节点集合。仅在 record_activated 为 True 时有效。
)doc")

            .def("get_activation_frequency",
                &pynetim::SusceptibleInfectedRecoveredModel::get_activation_frequency,
                R"doc(get_activation_frequency() -> list[int]

获取各节点的感染频数。

Returns:
    List[int]: 感染频数列表，索引 i 表示节点 i 被感染的次数。
)doc")

            .def("run_monte_carlo_diffusion",
                [](pynetim::SusceptibleInfectedRecoveredModel& self, int rounds, py::object seed_obj, bool use_multithread = false, int num_threads = 0) {
                    bool use_random_seed = seed_obj.is_none();
                    unsigned int seed = use_random_seed ? 0 : py::cast<unsigned int>(seed_obj);
                    return self.run_monte_carlo_diffusion(rounds, use_random_seed, seed, use_multithread, num_threads);
                },
                py::arg("rounds"),
                py::arg("seed") = py::none(),
                py::arg("use_multithread") = false,
                py::arg("num_threads") = 0,
                R"doc(run_monte_carlo_diffusion(rounds: int, seed: int | None = None, use_multithread: bool = False, num_threads: int = 0) -> float

运行蒙特卡洛模拟，计算平均影响力。

Args:
    rounds: 模拟次数，建议 1000-10000 次。
    seed: 随机种子，用于结果可重现。若为 None 则使用真随机种子。
    use_multithread: 是否启用多线程，默认为 False。
    num_threads: 线程数，0 表示自动检测。

Returns:
    float: 平均感染节点数。
)doc");
    }
}
