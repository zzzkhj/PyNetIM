"""
PyNetIM C++ 图测试

该文件包含 PyNetIM C++ 后端的完整测试用例，涵盖了：
- C++ 图的基本操作
- 邻接查询
- 度数查询
- 边权重设置
- 扩散模型测试（IC、LT、SI、SIR）

运行方式:
    conda run -n pynetim python tests/test_cpp.py
"""

import networkx as nx
from pynetim.cpp.graph import IMGraphCpp
from pynetim.cpp.diffusion_model import (
    IndependentCascadeModel,
    LinearThresholdModel,
    SusceptibleInfectedModel,
    SusceptibleInfectedRecoveredModel
)
from pynetim.py.algorithms import DegreeDiscountAlgorithm


def test_cpp_graph_basic():
    """测试 C++ 图的基本操作"""
    print("=" * 60)
    print("测试 1: C++ 图基本操作")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.erdos_renyi_graph(100, 0.2, seed=42, directed=True)
    
    edges = list(nx_graph.edges())
    weights = [1.0 / nx_graph.in_degree(v) for u, v in nx_graph.edges()]
    
    graph = IMGraphCpp(nx_graph.number_of_nodes(), edges, weights, nx_graph.is_directed())
    print(f"C++ 图: {graph}")
    print(f"  节点数: {graph.num_nodes}")
    print(f"  边数: {graph.num_edges}")
    print(f"  是否有向: {graph.directed}")


def test_cpp_neighbor_queries():
    """测试 C++ 图的邻接查询"""
    print("\n" + "=" * 60)
    print("测试 2: C++ 图邻接查询")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.erdos_renyi_graph(100, 0.2, seed=42, directed=True)
    
    edges = list(nx_graph.edges())
    weights = [1.0 / nx_graph.in_degree(v) for u, v in nx_graph.edges()]
    
    graph = IMGraphCpp(nx_graph.number_of_nodes(), edges, weights, nx_graph.is_directed())
    
    print("\n测试节点 0-9:")
    for i in range(10):
        out_neighbors = graph.out_neighbors(i)
        in_neighbors = graph.in_neighbors(i)
        out_degree = graph.out_degree(i)
        in_degree = graph.in_degree(i)
        print(f"  节点 {i}: 出度={out_degree}, 入度={in_degree}")
        print(f"    出邻居: {[(e.to, e.weight) for e in out_neighbors]}")
        print(f"    入邻居: {list(in_neighbors)}")


def test_cpp_degree_queries():
    """测试 C++ 图的度数查询"""
    print("\n" + "=" * 60)
    print("测试 3: C++ 图度数查询")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.erdos_renyi_graph(100, 0.2, seed=42, directed=True)
    
    edges = list(nx_graph.edges())
    weights = [1.0 / nx_graph.in_degree(v) for u, v in nx_graph.edges()]
    
    graph = IMGraphCpp(nx_graph.number_of_nodes(), edges, weights, nx_graph.is_directed())
    
    print("\n批量查询度数:")
    test_nodes = list(range(0, 100, 10))
    
    out_degrees = graph.batch_out_degree(test_nodes)
    in_degrees = graph.batch_in_degree(test_nodes)
    degrees = graph.batch_degree(test_nodes)
    
    print(f"  出度: {out_degrees[:5]}")
    print(f"  入度: {in_degrees[:5]}")
    print(f"  总度: {degrees[:5]}")


def test_cpp_edge_weights():
    """测试 C++ 图的边权重设置"""
    print("\n" + "=" * 60)
    print("测试 4: C++ 图边权重设置")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.erdos_renyi_graph(100, 0.2, seed=42, directed=True)
    
    edges = list(nx_graph.edges())
    weights = [1.0 / nx_graph.in_degree(v) for u, v in nx_graph.edges()]
    
    graph = IMGraphCpp(nx_graph.number_of_nodes(), edges, weights, nx_graph.is_directed())
    
    print("\n原始边权重:")
    for i in range(min(5, len(edges))):
        u, v = edges[i]
        print(f"  ({u}, {v}): {graph.get_edge_weight(u, v):.6f}")
    
    print("\n设置 CONSTANT 权重 (1.5):")
    for u, v in edges[:10]:
        graph.update_edge_weight(u, v, 1.5)
    print(f"  设置 ({u}, {v}): 1.5")
    
    print("\n更新后的边权重:")
    for i in range(min(5, len(edges))):
        u, v = edges[i]
        print(f"  ({u}, {v}): {graph.get_edge_weight(u, v):.6f}")


def test_cpp_ic_model():
    """测试 IC 扩散模型"""
    print("\n" + "=" * 60)
    print("测试 5: IC 扩散模型")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.erdos_renyi_graph(100, 0.2, seed=42, directed=True)
    
    edges = list(nx_graph.edges())
    weights = [random.random() for _ in edges]
    
    graph = IMGraphCpp(nx_graph.number_of_nodes(), edges, weights, nx_graph.is_directed())
    
    seeds = [0, 1, 2, 3, 4]
    
    ic_model = IndependentCascadeModel(graph, seeds)
    print(f"  种子节点: {seeds}")
    
    result = ic_model.diffusion()
    print(f"  激活节点数: {len(result)}")


def test_cpp_lt_model():
    """测试 LT 扩散模型"""
    print("\n" + "=" * 60)
    print("测试 6: LT 扩散模型")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.erdos_renyi_graph(100, 0.2, seed=42, directed=True)
    
    edges = list(nx_graph.edges())
    weights = [random.random() for _ in edges]
    
    graph = IMGraphCpp(nx_graph.number_of_nodes(), edges, weights, nx_graph.is_directed())
    
    seeds = [0, 1, 2, 3, 4]
    
    lt_model = LinearThresholdModel(graph, seeds)
    print(f"  种子节点: {seeds}")
    
    result = lt_model.diffusion()
    print(f"  激活节点数: {len(result)}")


def test_cpp_si_model():
    """测试 SI 扩散模型"""
    print("\n" + "=" * 60)
    print("测试 7: SI 扩散模型")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.erdos_renyi_graph(100, 0.2, seed=42, directed=True)
    
    edges = list(nx_graph.edges())
    weights = [random.random() for _ in edges]
    
    graph = IMGraphCpp(nx_graph.number_of_nodes(), edges, weights, nx_graph.is_directed())
    
    seeds = [0]
    
    si_model = SusceptibleInfectedModel(graph, seeds, beta=0.1)
    print(f"  初始感染节点: {seeds}")
    
    result = si_model.diffusion(2)
    print(f"  感染节点数: {len(result)}")


def test_cpp_sir_model():
    """测试 SIR 扩散模型"""
    print("\n" + "=" * 60)
    print("测试 8: SIR 扩散模型")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.erdos_renyi_graph(100, 0.2, seed=42, directed=True)
    
    edges = list(nx_graph.edges())
    weights = [random.random() for _ in edges]
    
    graph = IMGraphCpp(nx_graph.number_of_nodes(), edges, weights, nx_graph.is_directed())
    
    seeds = [0]
    
    sir_model = SusceptibleInfectedRecoveredModel(graph, seeds, gamma=0.5, beta=0.1)
    print(f"  初始感染节点: {seeds}")
    
    result = sir_model.diffusion(3)
    print(f"  感染节点数: {len(result)}")


def run_all_cpp_tests():
    """运行所有 C++ 图测试"""
    print("\n" + "=" * 60)
    print("开始运行 PyNetIM C++ 图测试套件")
    print("=" * 60)
    print()
    
    try:
        test_cpp_graph_basic()
        test_cpp_neighbor_queries()
        test_cpp_degree_queries()
        test_cpp_edge_weights()
        test_cpp_ic_model()
        test_cpp_lt_model()
        test_cpp_si_model()
        test_cpp_sir_model()
        
        print("\n" + "=" * 60)
        print("✅ 所有 C++ 图测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)


if __name__ == "__main__":
    run_all_cpp_tests()