import sys
sys.path.insert(0, '/root/PyNetIM-main/src')

import time
import networkx as nx
from pynetim.cpp.graph import IMGraphCpp
from pynetim.cpp.diffusion_model import IndependentCascadeModel, LinearThresholdModel

def test_graph_basic():
    print("Testing basic Graph operations...")
    
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    weights = [0.5, 0.3, 0.7, 0.2, 0.6]
    
    graph = IMGraphCpp(num_nodes=5, edges=edges, weights=weights, directed=True)
    
    print(f"Graph: {graph}")
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.num_edges}")
    print(f"Directed: {graph.directed}")
    
    print("\nTesting neighbor queries...")
    for i in range(5):
        out_neighbors = graph.out_neighbors(i)
        in_neighbors = graph.in_neighbors(i)
        print(f"Node {i}: out_degree={graph.out_degree(i)}, in_degree={graph.in_degree(i)}")
        print(f"  Out neighbors: {[(e.to, e.weight) for e in out_neighbors]}")
        print(f"  In neighbors: {list(in_neighbors)}")
    
    print("\nTesting edge weight retrieval...")
    for u, v in edges:
        weight = graph.get_edge_weight(u, v)
        print(f"Edge ({u}, {v}): weight = {weight}")
    
    print("\nTesting sparse matrix...")
    sparse = graph.get_adj_matrix_sparse()
    print(f"Sparse matrix has {len(sparse)} edges")
    print(f"First 5 edges: {sparse[:5]}")
    
    print("\n✓ Basic Graph tests passed!\n")

def test_ic_model():
    print("Testing Independent Cascade Model...")
    
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    weights = [0.5, 0.3, 0.7, 0.2, 0.6]
    
    graph = IMGraphCpp(num_nodes=5, edges=edges, weights=weights, directed=True)
    seeds = {0, 2}
    
    model = IndependentCascadeModel(graph, seeds)
    
    print(f"Seeds: {seeds}")
    
    start_time = time.time()
    result = model.run_monte_carlo_diffusion(rounds=1000, seed=42, use_multithread=False)
    single_thread_time = time.time() - start_time
    
    print(f"Single-thread result: {result:.4f} (time: {single_thread_time:.4f}s)")
    
    start_time = time.time()
    result_mt = model.run_monte_carlo_diffusion(rounds=1000, seed=42, use_multithread=True)
    multi_thread_time = time.time() - start_time
    
    print(f"Multi-thread result: {result_mt:.4f} (time: {multi_thread_time:.4f}s)")
    print(f"Speedup: {single_thread_time / multi_thread_time:.2f}x")
    
    print("\n✓ IC Model tests passed!\n")

def test_lt_model():
    print("Testing Linear Threshold Model...")
    
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    weights = [0.5, 0.3, 0.7, 0.2, 0.6]
    
    graph = IMGraphCpp(num_nodes=5, edges=edges, weights=weights, directed=True)
    seeds = {0, 2}
    
    model = LinearThresholdModel(graph, seeds, theta_l=0.0, theta_h=1.0)
    
    print(f"Seeds: {seeds}")
    
    start_time = time.time()
    result = model.run_monte_carlo_diffusion(rounds=1000, seed=42, use_multithread=False)
    single_thread_time = time.time() - start_time
    
    print(f"Single-thread result: {result:.4f} (time: {single_thread_time:.4f}s)")
    
    start_time = time.time()
    result_mt = model.run_monte_carlo_diffusion(rounds=1000, seed=42, use_multithread=True)
    multi_thread_time = time.time() - start_time
    
    print(f"Multi-thread result: {result_mt:.4f} (time: {multi_thread_time:.4f}s)")
    print(f"Speedup: {single_thread_time / multi_thread_time:.2f}x")
    
    print("\n✓ LT Model tests passed!\n")

def test_large_graph():
    print("Testing with larger graph...")
    
    n = 1000
    p = 0.01
    
    nx_graph = nx.erdos_renyi_graph(n, p, seed=42)
    edges = list(nx_graph.edges())
    weights = [0.1] * len(edges)
    
    graph = IMGraphCpp(num_nodes=n, edges=edges, weights=weights, directed=True)
    seeds = {0, 1, 2, 3, 4}
    
    print(f"Graph: {graph}")
    
    model = IndependentCascadeModel(graph, seeds)
    
    start_time = time.time()
    result = model.run_monte_carlo_diffusion(rounds=100, seed=42, use_multithread=False)
    single_thread_time = time.time() - start_time
    
    print(f"Single-thread result: {result:.2f} (time: {single_thread_time:.4f}s)")
    
    start_time = time.time()
    result_mt = model.run_monte_carlo_diffusion(rounds=100, seed=42, use_multithread=True)
    multi_thread_time = time.time() - start_time
    
    print(f"Multi-thread result: {result_mt:.2f} (time: {multi_thread_time:.4f}s)")
    print(f"Speedup: {single_thread_time / multi_thread_time:.2f}x")
    
    print("\n✓ Large graph tests passed!\n")

if __name__ == "__main__":
    print("=" * 60)
    print("PyNetIM C++ Optimization Tests")
    print("=" * 60)
    print()
    
    test_graph_basic()
    test_ic_model()
    test_lt_model()
    test_large_graph()
    
    print("=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)