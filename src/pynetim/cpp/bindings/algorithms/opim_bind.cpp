#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "algorithms/opim_algorithm.h"
#include "graph/Graph.h"

namespace py = pybind11;

PYBIND11_MODULE(opim_algorithm, m) {
    m.doc() = "OPIM算法模块，提供Online Processing for Influence Maximization算法";

    py::class_<pynetim::OPIMAlgorithm, pynetim::BaseRISAlgorithm, std::shared_ptr<pynetim::OPIMAlgorithm>>(m, "OPIMAlgorithm")
        .def(py::init([](std::shared_ptr<pynetim::Graph> graph,
                         const std::string& model,
                         std::optional<int> random_seed,
                         bool verbose) {
            return std::make_shared<pynetim::OPIMAlgorithm>(graph, model, random_seed, verbose);
        }),
            py::arg("graph"),
            py::arg("model"),
            py::arg("random_seed") = py::none(),
            py::arg("verbose") = false,
            R"doc(__init__(graph: IMGraph, model: str, random_seed: int | None = None, verbose: bool = False) -> None

初始化OPIM算法。

OPIM (Online Processing for Influence Maximization) 使用两组独立的RR集合：
- R1用于贪心选择种子节点
- R2用于验证选中种子的影响力

参考文献:
    Jing Tang, Xueyan Tang, Xiaokui Xiao, Junsong Yuan, 
    "Online Processing Algorithms for Influence Maximization," 
    in Proc. ACM SIGMOD, 2018.

Args:
    graph: 图对象。
    model: 扩散模型，支持 'IC' 或 'LT'。
    random_seed: 随机种子，默认为 None（每次随机）。
    verbose: 是否显示关键过程日志，默认为 False。
)doc")

        .def("run", &pynetim::OPIMAlgorithm::run,
            py::arg("k"),
            py::arg("num_rr_sets"),
            py::arg("delta") = -1.0,
            R"doc(run(k: int, num_rr_sets: int, delta: float = -1.0) -> set[int]

执行OPIM算法选择种子节点。

OPIM使用固定数量的RR集合，训练集和验证集使用相同大小。

Args:
    k: 需要选择的种子节点数量。
    num_rr_sets: 训练集和验证集各使用的RR集合数量。
    delta: 失败概率参数，默认为 1/n。

Returns:
    set[int]: 选中的种子节点集合。
)doc")

        .def("get_seeds", &pynetim::OPIMAlgorithm::getSeeds,
            R"doc(get_seeds() -> set[int]

获取最后一次运行选出的种子集合。

Returns:
    set[int]: 种子节点集合。
)doc")

        .def("get_approximation", &pynetim::OPIMAlgorithm::getApproximation,
            R"doc(get_approximation() -> float

获取最后一次运行的近似保证值。

Returns:
    float: 近似保证值 α。
)doc")

        .def("get_influence", &pynetim::OPIMAlgorithm::getInfluence,
            R"doc(get_influence() -> float

获取最后一次运行的影响力估计值（通过R2验证集）。

Returns:
    float: 影响力估计值。
)doc");

    py::class_<pynetim::OPIMCAlgorithm, pynetim::OPIMAlgorithm,
               std::shared_ptr<pynetim::OPIMCAlgorithm>>(m, "OPIMCAlgorithm")
        .def(py::init([](std::shared_ptr<pynetim::Graph> graph,
                         const std::string& model,
                         std::optional<int> random_seed,
                         bool verbose) {
            return std::make_shared<pynetim::OPIMCAlgorithm>(graph, model, random_seed, verbose);
        }),
            py::arg("graph"),
            py::arg("model"),
            py::arg("random_seed") = py::none(),
            py::arg("verbose") = false,
            R"doc(__init__(graph: IMGraph, model: str, random_seed: int | None = None, verbose: bool = False) -> None

初始化OPIM-C算法。

OPIM-C是OPIM的自适应版本，迭代增加RR集合数量，
直到达到(1-1/e-ε)近似保证。

参考文献:
    Jing Tang, Xueyan Tang, Xiaokui Xiao, Junsong Yuan, 
    "Online Processing Algorithms for Influence Maximization," 
    in Proc. ACM SIGMOD, 2018.

Args:
    graph: 图对象。
    model: 扩散模型，支持 'IC' 或 'LT'。
    random_seed: 随机种子，默认为 None（每次随机）。
    verbose: 是否显示关键过程日志，默认为 False。
)doc")

        .def("run", &pynetim::OPIMCAlgorithm::run,
            py::arg("k"),
            py::arg("epsilon"),
            py::arg("delta") = -1.0,
            R"doc(run(k: int, epsilon: float, delta: float = -1.0) -> set[int]

执行OPIM-C算法选择种子节点。

OPIM-C会自动迭代增加RR集合数量，直到达到目标近似保证。

Args:
    k: 需要选择的种子节点数量。
    epsilon: 误差阈值，算法返回(1-1/e-ε)-近似解。
    delta: 失败概率参数，默认为 1/n。

Returns:
    set[int]: 选中的种子节点集合。
)doc");
}
