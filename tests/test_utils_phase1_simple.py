import random
import networkx as nx
from pynetim.utils import graph_statistics, graph_density, connectivity_analysis

def test_graph_statistics():
    print("=" * 60)
    print("Testing graph_statistics Function")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.DiGraph()
    for _ in range(500):
        u = random.randint(0, 99)
        v = random.randint(0, 99)
        if u != v:
            nx_graph.add_edge(u, v)
    
    print("\nNetworkX Graph:")
    nx_stats = graph_statistics(nx_graph)
    for key, value in nx_stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("All graph_statistics tests passed!")
    print("=" * 60)

def test_graph_density():
    print("\n" + "=" * 60)
    print("Testing graph_density Function")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.DiGraph()
    for _ in range(500):
        u = random.randint(0, 99)
        v = random.randint(0, 99)
        if u != v:
            nx_graph.add_edge(u, v)
    
    print("\nNetworkX Graph:")
    nx_density = graph_density(nx_graph)
    print(f"  Density: {nx_density:.6f}")
    
    print("\n" + "=" * 60)
    print("All graph_density tests passed!")
    print("=" * 60)

def test_connectivity_analysis():
    print("\n" + "=" * 60)
    print("Testing connectivity_analysis Function")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.DiGraph()
    for _ in range(500):
        u = random.randint(0, 99)
        v = random.randint(0, 99)
        if u != v:
            nx_graph.add_edge(u, v)
    
    print("\nNetworkX Graph:")
    nx_conn = connectivity_analysis(nx_graph)
    for key, value in nx_conn.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("All connectivity_analysis tests passed!")
    print("=" * 60)

def test_disconnected_graph():
    print("\n" + "=" * 60)
    print("Testing with Disconnected Graph")
    print("=" * 60)
    
    random.seed(42)
    
    nx_graph = nx.DiGraph()
    for i in range(50):
        for j in range(i + 1, 50):
            nx_graph.add_edge(i, j)
    
    print("\nNetworkX Graph (disconnected):")
    nx_stats = graph_statistics(nx_graph)
    nx_conn = connectivity_analysis(nx_graph)
    
    print(f"  is_connected: {nx_stats['is_connected']}")
    print(f"  num_components: {nx_stats['num_components']}")
    print(f"  largest_component_size: {nx_conn['largest_component_size']}")
    print(f"  component_sizes: {nx_conn['component_sizes'][:5]}...")

def test_empty_graph():
    print("\n" + "=" * 60)
    print("Testing with Empty Graph")
    print("=" * 60)
    
    nx_graph = nx.DiGraph()
    
    print("\nEmpty NetworkX Graph:")
    nx_stats = graph_statistics(nx_graph)
    for key, value in nx_stats.items():
        print(f"  {key}: {value}")

def run_all_tests():
    print("=" * 60)
    print("Utils Module Phase 1 Tests")
    print("=" * 60)
    
    test_graph_statistics()
    test_graph_density()
    test_connectivity_analysis()
    test_disconnected_graph()
    test_empty_graph()
    
    print("\n" + "=" * 60)
    print("All Phase 1 tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
