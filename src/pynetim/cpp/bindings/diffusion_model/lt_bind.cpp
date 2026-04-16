#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "diffusion_model/linear_threshold.h"

namespace py = pybind11;

PYBIND11_MODULE(linear_threshold_model, m) {
    m.doc() = "线性阈值模型（LT），用于社交网络影响力传播模拟";

    {
        py::options options;
        options.disable_function_signatures();

        py::class_<pynetim::LinearThresholdModel>(m, "LinearThresholdModel")
            .def(py::init([](py::object graph_obj, const std::set<int>& seeds, double theta_l, double theta_h, bool record_activated, bool record_activation_frequency) {
                try {
                    auto graph_ptr = py::cast<std::shared_ptr<pynetim::Graph>>(graph_obj);
                    return std::make_unique<pynetim::LinearThresholdModel>(graph_ptr, seeds, theta_l, theta_h, record_activated, record_activation_frequency);
                } catch (const py::cast_error&) {
                    throw py::type_error("LinearThresholdModel() 参数错误: graph 必须是 IMGraph 类型。\n用法: LinearThresholdModel(graph, seeds, theta_l=0.0, theta_h=1.0, record_activated=False, record_activation_frequency=False)");
                }
            }),
                py::arg("graph"), py::arg("seeds"), py::arg("theta_l") = 0.0, py::arg("theta_h") = 1.0, py::arg("record_activated") = false, py::arg("record_activation_frequency") = false,
                py::keep_alive<0, 1>(),
                R"doc(__init__(graph: IMGraph, seeds: set[int], theta_l: float = 0.0, theta_h: float = 1.0, record_activated: bool = False, record_activation_frequency: bool = False) -> None

构造线性阈值模型。

Args:
    graph: 图对象。
    seeds: 初始种子节点集合。
    theta_l: 阈值下界，默认为 0.0。
    theta_h: 阈值上界，默认为 1.0。
    record_activated: 是否记录激活节点，默认为 False。
    record_activation_frequency: 是否记录激活频数，默认为 False。
)doc")

            .def("set_seeds", &pynetim::LinearThresholdModel::set_seeds,
                py::arg("new_seeds"),
                R"doc(set_seeds(new_seeds: set[int]) -> None

更新种子节点集合。

Args:
    new_seeds: 新的种子节点集合。
)doc")

            .def("set_record_activated", &pynetim::LinearThresholdModel::set_record_activated,
                py::arg("record"),
                R"doc(set_record_activated(record: bool) -> None

启用或禁用激活节点记录。

Args:
    record: 是否记录激活节点。
)doc")

            .def("set_record_activation_frequency", &pynetim::LinearThresholdModel::set_record_activation_frequency,
                py::arg("record"),
                R"doc(set_record_activation_frequency(record: bool) -> None

启用或禁用激活频数记录。

Args:
    record: 是否记录激活频数。
)doc")

            .def("run_single_simulation",
                [](pynetim::LinearThresholdModel& self, py::object random_seed_obj) {
                    bool use_random_seed = random_seed_obj.is_none();
                    unsigned int random_seed = use_random_seed ? 0 : py::cast<unsigned int>(random_seed_obj);
                    return self.run_single_simulation(use_random_seed, random_seed);
                },
                py::arg("random_seed") = py::none(),
                R"doc(run_single_simulation(random_seed: int | None = None) -> int

执行单次传播模拟。

Args:
    random_seed: 随机种子，用于结果可重现。若为 None 则使用真随机种子。

Returns:
    int: 本次模拟激活的节点数。
)doc")

            .def("get_activated_nodes",
                &pynetim::LinearThresholdModel::get_activated_nodes,
                R"doc(get_activated_nodes() -> set[int]

获取上次模拟的激活节点集合。

Returns:
    Set[int]: 激活节点集合。仅在 record_activated 为 True 时有效。
)doc")

            .def("get_activation_frequency",
                &pynetim::LinearThresholdModel::get_activation_frequency,
                R"doc(get_activation_frequency() -> list[int]

获取各节点的激活频数。

Returns:
    List[int]: 激活频数列表，索引 i 表示节点 i 被激活的次数。
)doc")

            .def("run_monte_carlo_diffusion",
                [](pynetim::LinearThresholdModel& self, int mc_rounds, py::object random_seed_obj, bool use_multithread = false, int num_threads = 0) {
                    if (use_multithread && num_threads <= 0) {
                        throw std::invalid_argument("启用多线程时，线程数(num_threads)必须大于0");
                    }
                    bool use_random_seed = random_seed_obj.is_none();
                    unsigned int random_seed = use_random_seed ? 0 : py::cast<unsigned int>(random_seed_obj);
                    return self.run_monte_carlo_diffusion(mc_rounds, use_random_seed, random_seed, use_multithread, num_threads);
                },
                py::arg("mc_rounds"),
                py::arg("random_seed") = py::none(),
                py::arg("use_multithread") = false,
                py::arg("num_threads") = 0,
                R"doc(run_monte_carlo_diffusion(mc_rounds: int, random_seed: int | None = None, use_multithread: bool = False, num_threads: int = 0) -> float

运行蒙特卡洛模拟，计算平均影响力。

Args:
    mc_rounds: 蒙特卡洛模拟次数，建议 1000-10000 次。
    random_seed: 随机种子，用于结果可重现。若为 None 则使用真随机种子。
    use_multithread: 是否启用多线程，默认为 False。
    num_threads: 线程数，当 use_multithread=True 时必须大于 0。

Returns:
    float: 平均激活节点数。

Raises:
    ValueError: 当 use_multithread=True 但 num_threads <= 0 时抛出。
)doc");
    }
}
