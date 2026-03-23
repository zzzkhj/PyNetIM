"""
PyNetIM 测试套件

该文件包含 PyNetIM 库的完整测试用例，涵盖了：
- NetworkX 图的基本操作
- 图统计信息计算
- 图密度计算
- 图连通性分析
- 工具函数测试

运行方式:
    conda run -n pynetim python tests/test.py
"""

import sys
import networkx as nx
from pynetim.utils import graph_statistics, graph_density, connectivity_analysis, set_edge_weight, infection_threshold


def test_basic_graph_operations():
    """测试基本的图操作"""
    print("=" * 60)
    print("测试 1: 基本图操作")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.DiGraph()
    for _ in range(500):
        u = random.randint(0, 99)
        v = random.randint(0, 99)
        if u != v:
            nx_graph.add_edge(u, v)
    
    print(f"创建的图: {nx_graph.number_of_nodes()} 个节点, {nx_graph.number_of_edges()} 条边")


def test_graph_statistics():
    """测试图统计信息函数"""
    print("\n" + "=" * 60)
    print("测试 2: 图统计信息")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.DiGraph()
    for _ in range(500):
        u = random.randint(0, 99)
        v = random.randint(0, 99)
        if u != v:
            nx_graph.add_edge(u, v)
    
    print("\nNetworkX 图:")
    stats = graph_statistics(nx_graph)
    for key, value in stats.items():
        print(f"  {key}: {value}")


def test_graph_density():
    """测试图密度计算函数"""
    print("\n" + "=" * 60)
    print("测试 3: 图密度计算")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.DiGraph()
    for _ in range(500):
        u = random.randint(0, 99)
        v = random.randint(0, 99)
        if u != v:
            nx_graph.add_edge(u, v)
    
    print("\nNetworkX Graph:")
    density = graph_density(nx_graph)
    print(f"  密度: {density:.6f}")


def test_connectivity_analysis():
    """测试连通性分析函数"""
    print("\n" + "=" * 60)
    print("测试 4: 连通性分析")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.DiGraph()
    for _ in range(500):
        u = random.randint(0, 99)
        v = random.randint(0, 99)
        if u != v:
            nx_graph.add_edge(u, v)
    
    print("\nNetworkX Graph:")
    connectivity = connectivity_analysis(nx_graph)
    for key, value in connectivity.items():
        print(f"  {key}: {value}")


def test_edge_weight_setting():
    """测试边权重设置"""
    print("\n" + "=" * 60)
    print("测试 5: 边权重设置")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.DiGraph()
    for _ in range(500):
        u = random.randint(0, 99)
        v = random.randint(0, 99)
        if u != v:
            nx_graph.add_edge(u, v)
    
    print("\n测试 CONSTANT 权重:")
    set_edge_weight(nx_graph, 'CONSTANT', constant_weight=1.0)
    print(f"  边权重: {[(u, v, nx_graph.edges[(u, v)]['weight']) for u, v in list(nx_graph.edges())[:5]]}")
    
    print("\n测试 TV 权重:")
    set_edge_weight(nx_graph, 'TV')
    print(f"  边权重: {[(u, v, nx_graph.edges[(u, v)]['weight']) for u, v in list(nx_graph.edges())[:5]]}")
    
    print("\n测试 WC 权重:")
    set_edge_weight(nx_graph, 'WC')
    print(f"  边权重: {[(u, v, nx_graph.edges[(u, v)]['weight']) for u, v in list(nx_graph.edges())[:5]]}")


def test_infection_threshold():
    """测试感染阈值计算"""
    print("\n" + "=" * 60)
    print("测试 6: 感染阈值计算")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.DiGraph()
    for _ in range(500):
        u = random.randint(0, 99)
        v = random.randint(0, 99)
        if u != v:
            nx_graph.add_edge(u, v)
    
    print("\nNetworkX Graph:")
    threshold = infection_threshold(nx_graph)
    print(f"  感染阈值: {threshold:.6f}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始运行 PyNetIM 测试套件")
    print("=" * 60)
    print()
    
    try:
        test_basic_graph_operations()
        test_graph_statistics()
        test_graph_density()
        test_connectivity_analysis()
        test_edge_weight_setting()
        test_infection_threshold()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()