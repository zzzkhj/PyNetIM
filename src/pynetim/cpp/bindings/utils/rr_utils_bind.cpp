#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/rr_utils.h"
#include "../include/graph/Graph.h"

namespace py = pybind11;

PYBIND11_MODULE(rr_utils, m) {
    m.doc() = "RR 集合生成工具函数";

    m.def("sample_rr_set_ic", [](
        std::shared_ptr<pynetim::Graph> graph,
        py::object seed_obj
    ) -> std::vector<int> {
        std::mt19937 rng;
        if (!seed_obj.is_none()) {
            rng.seed(seed_obj.cast<int>());
        } else {
            std::random_device rd;
            rng.seed(rd());
        }
        return pynetim::utils::sampleRRSetIC(graph, rng);
    }, py::arg("graph"), py::arg("seed") = py::none(),
        R"doc(
采样一个 IC 模型的 RR 集合。

Args:
    graph: IMGraph 图对象。
    seed: 随机种子，可选。

Returns:
    List[int]: RR 集合中的节点列表。

Example:
    >>> from pynetim.utils import sample_rr_set_ic
    >>> rr_set = sample_rr_set_ic(graph, seed=42)
)doc");

    m.def("sample_rr_set_lt", [](
        std::shared_ptr<pynetim::Graph> graph,
        py::object seed_obj
    ) -> std::vector<int> {
        std::mt19937 rng;
        if (!seed_obj.is_none()) {
            rng.seed(seed_obj.cast<int>());
        } else {
            std::random_device rd;
            rng.seed(rd());
        }
        return pynetim::utils::sampleRRSetLT(graph, rng);
    }, py::arg("graph"), py::arg("seed") = py::none(),
        R"doc(
采样一个 LT 模型的 RR 集合。

Args:
    graph: IMGraph 图对象。
    seed: 随机种子，可选。

Returns:
    List[int]: RR 集合中的节点列表。

Example:
    >>> from pynetim.utils import sample_rr_set_lt
    >>> rr_set = sample_rr_set_lt(graph, seed=42)
)doc");

    m.def("generate_rr_sets", &pynetim::utils::generateRRSets,
        py::arg("graph"), py::arg("num_sets"), py::arg("model") = "IC", py::arg("seed") = py::none(),
        R"doc(
生成多个 RR 集合。

Args:
    graph: IMGraph 图对象。
    num_sets: 要生成的 RR 集合数量。
    model: 传播模型，支持 'IC' 或 'LT'，默认为 'IC'。
    seed: 随机种子，可选。

Returns:
    List[List[int]]: RR 集合列表。

Example:
    >>> from pynetim.utils import generate_rr_sets
    >>> rr_sets = generate_rr_sets(graph, 1000, model='IC', seed=42)
)doc");
}
