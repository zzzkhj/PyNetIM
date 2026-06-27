#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "algorithms/imm_algorithm.h"
#include "graph/Graph.h"

namespace py = pybind11;

namespace {

std::optional<int> resolve_random_seed_opt(py::object random_seed_obj) {
    if (!random_seed_obj.is_none()) {
        return py::cast<int>(random_seed_obj);
    }
    try {
        py::module_ random_module = py::module_::import("pynetim.random");
        py::object seed_obj = random_module.attr("get_random_seed")();
        if (!seed_obj.is_none()) {
            return py::cast<int>(seed_obj);
        }
    } catch (...) {
    }
    return std::nullopt;
}

}

PYBIND11_MODULE(imm_algorithm, m) {
    m.doc() = "IMM算法模块";

    {
        py::options options;
        options.disable_function_signatures();

        py::class_<pynetim::IMMAlgorithm, pynetim::BaseRISAlgorithm,
                   std::shared_ptr<pynetim::IMMAlgorithm>>(m, "IMMAlgorithm")
        .def(py::init([](py::object graph_obj,
                         const std::string& model,
                         double epsilon,
                         int l,
                         py::object random_seed_obj,
                         bool verbose) {
            std::shared_ptr<pynetim::Graph> graph_ptr;
            try {
                graph_ptr = py::cast<std::shared_ptr<pynetim::Graph>>(graph_obj);
            } catch (const py::cast_error&) {
                throw py::type_error("IMMAlgorithm() 参数错误: graph 必须是 IMGraph 类型。\n用法: IMMAlgorithm(graph, model, epsilon=0.5, l=1, random_seed=None, verbose=False)");
            }
            return std::make_shared<pynetim::IMMAlgorithm>(graph_ptr, model, epsilon, l, resolve_random_seed_opt(random_seed_obj), verbose);
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
    random_seed: 随机种子，默认为 None（使用全局种子或随机）。
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
    }
}
