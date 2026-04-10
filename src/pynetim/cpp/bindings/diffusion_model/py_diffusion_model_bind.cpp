#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "diffusion_model/py_diffusion_model_base.h"

namespace py = pybind11;
namespace pynetim {

class PyDiffusionModelBaseTrampoline : public PyDiffusionModelBase {
public:
    using PyDiffusionModelBase::PyDiffusionModelBase;

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

        try {
            py::object self = py::cast(this);
            py::object result = self.attr("run_single_trial")(
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
        } catch (const py::error_already_set&) {
            throw;
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            throw;
        }
    }
};

}

PYBIND11_MODULE(py_diffusion_model_base, m) {
    m.doc() = "Python 兼容的传播模型基类，用于自定义传播逻辑";

    {
        py::options options;
        options.disable_function_signatures();

        py::class_<pynetim::PyDiffusionModelBase, pynetim::PyDiffusionModelBaseTrampoline>(m, "PyDiffusionModelBase")
            .def(py::init([](py::object graph_obj, const std::set<int>& seeds,
                             py::object py_override, bool record_activated, bool record_activation_frequency) {
                auto graph_ptr = py::cast<std::shared_ptr<pynetim::Graph>>(graph_obj);
                return std::make_unique<pynetim::PyDiffusionModelBaseTrampoline>(
                    graph_ptr, seeds, py_override, record_activated, record_activation_frequency);
            }),
                py::arg("graph"),
                py::arg("seeds"),
                py::arg("py_override") = py::none(),
                py::arg("record_activated") = false,
                py::arg("record_activation_frequency") = false,
                py::keep_alive<0, 1>(),
                R"doc(__init__(graph: IMGraph, seeds: set[int], py_override: object = None, record_activated: bool = False, record_activation_frequency: bool = False) -> None

构造 Python 兼容的传播模型基类。

Args:
    graph: 图对象。
    seeds: 初始种子节点集合。
    py_override: Python 重写对象，默认为 None。
    record_activated: 是否记录激活节点，默认为 False。
    record_activation_frequency: 是否记录激活频数，默认为 False。
)doc")

            .def_property_readonly("graph", &pynetim::PyDiffusionModelBase::graph,
                R"doc(获取图对象。)doc")

            .def("set_seeds", &pynetim::PyDiffusionModelBase::set_seeds,
                py::arg("new_seeds"),
                R"doc(set_seeds(new_seeds: set[int]) -> None

更新种子节点集合。

Args:
    new_seeds: 新的种子节点集合。
)doc")

            .def("set_record_activated", &pynetim::PyDiffusionModelBase::set_record_activated,
                py::arg("record"),
                R"doc(set_record_activated(record: bool) -> None

启用或禁用激活节点记录。

Args:
    record: 是否记录激活节点。
)doc")

            .def("set_record_activation_frequency", &pynetim::PyDiffusionModelBase::set_record_activation_frequency,
                py::arg("record"),
                R"doc(set_record_activation_frequency(record: bool) -> None

启用或禁用激活频数记录。

Args:
    record: 是否记录激活频数。
)doc")

            .def("run_single_simulation",
                [](pynetim::PyDiffusionModelBase& self, py::object random_seed_obj) {
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
                [](const pynetim::PyDiffusionModelBase& self) {
                    return self.get_activated_nodes();
                },
                R"doc(get_activated_nodes() -> set[int]

获取上次模拟的激活节点集合。

Returns:
    Set[int]: 激活节点集合。
)doc")

            .def("get_activation_frequency",
                [](const pynetim::PyDiffusionModelBase& self) {
                    return self.get_activation_frequency();
                },
                R"doc(get_activation_frequency() -> list[int]

获取各节点的激活频数。

Returns:
    List[int]: 激活频数列表。
)doc")

            .def("run_monte_carlo_diffusion",
                [](pynetim::PyDiffusionModelBase& self, int mc_rounds, py::object random_seed_obj,
                   bool use_multithread, int num_threads) {
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

Note:
    多线程模式下受 Python GIL 限制，无法真正并行。
    建议使用 BaseMultiprocessDiffusionModel 实现真正的并行计算。
)doc")

            .def("set_py_override", &pynetim::PyDiffusionModelBase::set_py_override,
                py::arg("obj"),
                R"doc(set_py_override(obj: object) -> None

设置 Python 重写对象。

Args:
    obj: Python 重写对象，用于自定义传播逻辑。
)doc")

            .def("get_py_override", &pynetim::PyDiffusionModelBase::get_py_override,
                R"doc(get_py_override() -> object

获取 Python 重写对象。

Returns:
    object: Python 重写对象。
)doc");
    }
}
