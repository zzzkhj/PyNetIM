#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "algorithms/tim_algorithm.h"
#include "graph/Graph.h"

namespace py = pybind11;

PYBIND11_MODULE(tim_algorithm, m) {
    m.doc() = "TIM/TIM+算法模块";

    {
        py::options options;
        options.disable_function_signatures();

        py::class_<pynetim::TIMAlgorithm, pynetim::BaseRISAlgorithm,
                   std::shared_ptr<pynetim::TIMAlgorithm>>(m, "TIMAlgorithm")
        .def(py::init([](py::object graph_obj,
                         const std::string& model,
                         double epsilon,
                         int l,
                         std::optional<int> random_seed,
                         bool verbose) {
            std::shared_ptr<pynetim::Graph> graph_ptr;
            try {
                graph_ptr = py::cast<std::shared_ptr<pynetim::Graph>>(graph_obj);
            } catch (const py::cast_error&) {
                throw py::type_error("TIMAlgorithm() 参数错误: graph 必须是 IMGraph 类型。\n用法: TIMAlgorithm(graph, model, epsilon=0.5, l=1, random_seed=None, verbose=False)");
            }
            return std::make_shared<pynetim::TIMAlgorithm>(graph_ptr, model, epsilon, l, random_seed, verbose);
        }),
            py::arg("graph"),
            py::arg("model"),
            py::arg("epsilon") = 0.5,
            py::arg("l") = 1,
            py::arg("random_seed") = py::none(),
            py::arg("verbose") = false,
            R"doc(__init__(graph: IMGraph, model: str, epsilon: float = 0.5, l: int = 1, random_seed: int | None = None, verbose: bool = False) -> None

初始化TIM算法。

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

TIM+改进了TIM的采样策略，通常比TIM更快。

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

Args:
    k: 需要选择的种子节点数量。

Returns:
    set[int]: 选中的种子节点集合。
)doc");
    }
}
