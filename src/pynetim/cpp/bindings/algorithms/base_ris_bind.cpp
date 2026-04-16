#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "algorithms/base_ris_algorithm.h"
#include "graph/Graph.h"

namespace py = pybind11;

PYBIND11_MODULE(base_ris_algorithm, m) {
    m.doc() = "基础RIS算法模块";

    {
        py::options options;
        options.disable_function_signatures();

        py::class_<pynetim::BaseRISAlgorithm, std::shared_ptr<pynetim::BaseRISAlgorithm>>(m, "BaseRISAlgorithm")
        .def(py::init([](py::object graph_obj,
                         const std::string& model,
                         std::optional<int> random_seed,
                         bool verbose) {
            std::shared_ptr<pynetim::Graph> graph_ptr;
            try {
                graph_ptr = py::cast<std::shared_ptr<pynetim::Graph>>(graph_obj);
            } catch (const py::cast_error&) {
                throw py::type_error("BaseRISAlgorithm() 参数错误: graph 必须是 IMGraph 类型。\n用法: BaseRISAlgorithm(graph, model, random_seed=None, verbose=False)");
            }
            return std::make_shared<pynetim::BaseRISAlgorithm>(graph_ptr, model, random_seed, verbose);
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
    }
}
