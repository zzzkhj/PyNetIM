#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "ris_algorithm.h"
#include "Graph.h"

namespace py = pybind11;

PYBIND11_MODULE(ris_algorithm, m) {
    m.doc() = "RIS算法模块，提供基于反向影响采样的影响力最大化算法";

    py::class_<pynetim::BaseRISAlgorithm, std::shared_ptr<pynetim::BaseRISAlgorithm>>(m, "BaseRISAlgorithm")
        .def(py::init([](std::shared_ptr<pynetim::Graph> graph,
                         const std::string& model,
                         std::optional<int> random_seed,
                         bool verbose) {
            return std::make_shared<pynetim::BaseRISAlgorithm>(graph, model, random_seed, verbose);
        }),
            py::arg("graph"),
            py::arg("model"),
            py::arg("random_seed") = py::none(),
            py::arg("verbose") = false,
            R"doc(__init__(graph: IMGraph, model: str, random_seed: int | None = None, verbose: bool = False) -> None

初始化基础RIS算法。

Args:
    graph: 图对象。
    model: 扩散模型，支持 'IC' 或 'LT'。
    random_seed: 随机种子，默认为 None（每次随机）。
    verbose: 是否显示关键过程日志，默认为 False。
)doc")

        .def("run", &pynetim::BaseRISAlgorithm::run,
            py::arg("k"),
            py::arg("num_rr_sets"),
            R"doc(run(k: int, num_rr_sets: int) -> set[int]

执行RIS算法选择种子节点。

Args:
    k: 需要选择的种子节点数量。
    num_rr_sets: RR集合采样数量，越大越准确。

Returns:
    set[int]: 选中的种子节点集合。
)doc")

        .def("get_seeds", &pynetim::BaseRISAlgorithm::getSeeds,
            R"doc(get_seeds() -> set[int]

获取最后一次运行选出的种子集合。

Returns:
    set[int]: 种子节点集合。
)doc");

    py::class_<pynetim::IMMAlgorithm, pynetim::BaseRISAlgorithm,
               std::shared_ptr<pynetim::IMMAlgorithm>>(m, "IMMAlgorithm")
        .def(py::init([](std::shared_ptr<pynetim::Graph> graph,
                         const std::string& model,
                         double epsilon,
                         int l,
                         std::optional<int> random_seed,
                         bool verbose) {
            return std::make_shared<pynetim::IMMAlgorithm>(graph, model, epsilon, l, random_seed, verbose);
        }),
            py::arg("graph"),
            py::arg("model"),
            py::arg("epsilon") = 0.5,
            py::arg("l") = 1,
            py::arg("random_seed") = py::none(),
            py::arg("verbose") = false,
            R"doc(__init__(graph: IMGraph, model: str, epsilon: float = 0.5, l: int = 1, random_seed: int | None = None, verbose: bool = False) -> None

初始化IMM算法。

Args:
    graph: 图对象。
    model: 扩散模型，支持 'IC' 或 'LT'。
    epsilon: 近似参数ε，默认为 0.5。
    l: 失败概率参数，默认为 1。
    random_seed: 随机种子，默认为 None（每次随机）。
    verbose: 是否显示关键过程日志，默认为 False。
)doc")

        .def("run", py::overload_cast<int>(&pynetim::IMMAlgorithm::run),
            py::arg("k"),
            R"doc(run(k: int) -> set[int]

执行IMM算法选择种子节点。

IMM会自动估计最优采样数量，无需手动指定。

Args:
    k: 需要选择的种子节点数量。

Returns:
    set[int]: 选中的种子节点集合。
)doc");

    py::class_<pynetim::TIMAlgorithm, pynetim::BaseRISAlgorithm,
               std::shared_ptr<pynetim::TIMAlgorithm>>(m, "TIMAlgorithm")
        .def(py::init([](std::shared_ptr<pynetim::Graph> graph,
                         const std::string& model,
                         double epsilon,
                         int l,
                         std::optional<int> random_seed,
                         bool verbose) {
            return std::make_shared<pynetim::TIMAlgorithm>(graph, model, epsilon, l, random_seed, verbose);
        }),
            py::arg("graph"),
            py::arg("model"),
            py::arg("epsilon") = 0.5,
            py::arg("l") = 1,
            py::arg("random_seed") = py::none(),
            py::arg("verbose") = false,
            R"doc(__init__(graph: IMGraph, model: str, epsilon: float = 0.5, l: int = 1, random_seed: int | None = None, verbose: bool = False) -> None

初始化TIM算法。

TIM (Two-phase Influence Maximization) 是两阶段影响力最大化算法。

Args:
    graph: 图对象。
    model: 扩散模型，支持 'IC' 或 'LT'。
    epsilon: 近似参数ε，默认为 0.5。
    l: 失败概率参数，默认为 1。
    random_seed: 随机种子，默认为 None（每次随机）。
    verbose: 是否显示关键过程日志，默认为 False。
)doc")

        .def("run", py::overload_cast<int>(&pynetim::TIMAlgorithm::run),
            py::arg("k"),
            R"doc(run(k: int) -> set[int]

执行TIM算法选择种子节点。

TIM使用两阶段策略：第一阶段估计OPT，第二阶段采样并选择种子。

Args:
    k: 需要选择的种子节点数量。

Returns:
    set[int]: 选中的种子节点集合。
)doc");

    py::class_<pynetim::TIMPlusAlgorithm, pynetim::BaseRISAlgorithm,
               std::shared_ptr<pynetim::TIMPlusAlgorithm>>(m, "TIMPlusAlgorithm")
        .def(py::init([](std::shared_ptr<pynetim::Graph> graph,
                         const std::string& model,
                         double epsilon,
                         int l,
                         std::optional<int> random_seed,
                         bool verbose) {
            return std::make_shared<pynetim::TIMPlusAlgorithm>(graph, model, epsilon, l, random_seed, verbose);
        }),
            py::arg("graph"),
            py::arg("model"),
            py::arg("epsilon") = 0.5,
            py::arg("l") = 1,
            py::arg("random_seed") = py::none(),
            py::arg("verbose") = false,
            R"doc(__init__(graph: IMGraph, model: str, epsilon: float = 0.5, l: int = 1, random_seed: int | None = None, verbose: bool = False) -> None

初始化TIM+算法。

TIM+ 是TIM的改进版本，使用更高效的采样策略。

Args:
    graph: 图对象。
    model: 扩散模型，支持 'IC' 或 'LT'。
    epsilon: 近似参数ε，默认为 0.5。
    l: 失败概率参数，默认为 1。
    random_seed: 随机种子，默认为 None（每次随机）。
    verbose: 是否显示关键过程日志，默认为 False。
)doc")

        .def("run", py::overload_cast<int>(&pynetim::TIMPlusAlgorithm::run),
            py::arg("k"),
            R"doc(run(k: int) -> set[int]

执行TIM+算法选择种子节点。

TIM+改进了TIM的采样策略，通常比TIM更快。

Args:
    k: 需要选择的种子节点数量。

Returns:
    set[int]: 选中的种子节点集合。
)doc");
}
