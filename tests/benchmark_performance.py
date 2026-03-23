import time
import random
import numpy as np
from pynetim.cpp.graph import IMGraphCpp
from pynetim.cpp.diffusion_model import IndependentCascadeModel, LinearThresholdModel

def generate_random_graph(num_nodes, num_edges, directed=True, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    edges = []
    weights = []
    
    for _ in range(num_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v:
            edges.append((u, v))
            weights.append(random.random())
    
    return IMGraphCpp(num_nodes=num_nodes, edges=edges, weights=weights, directed=directed)

def benchmark_graph_operations(graph, num_runs=100):
    print("\n" + "=" * 60)
    print("Graph Operations Benchmark")
    print("=" * 60)
    
    results = {}
    
    print(f"\nGraph: {graph}")
    print(f"Number of runs: {num_runs}")
    
    test_nodes = list(range(min(100, graph.num_nodes)))
    
    start_time = time.time()
    for _ in range(num_runs):
        for u in test_nodes:
            graph.out_degree(u)
    elapsed = time.time() - start_time
    results['out_degree'] = elapsed / num_runs
    print(f"out_degree (individual): {results['out_degree']*1000:.4f} ms per run")
    
    start_time = time.time()
    for _ in range(num_runs):
        graph.batch_out_degree(test_nodes)
    elapsed = time.time() - start_time
    results['batch_out_degree'] = elapsed / num_runs
    print(f"batch_out_degree: {results['batch_out_degree']*1000:.4f} ms per run")
    print(f"Speedup: {results['out_degree'] / results['batch_out_degree']:.2f}x")
    
    start_time = time.time()
    for _ in range(num_runs):
        for u in test_nodes:
            graph.out_neighbors(u)
    elapsed = time.time() - start_time
    results['out_neighbors'] = elapsed / num_runs
    print(f"\nout_neighbors (individual): {results['out_neighbors']*1000:.4f} ms per run")
    
    start_time = time.time()
    for _ in range(num_runs):
        graph.batch_out_neighbors(test_nodes)
    elapsed = time.time() - start_time
    results['batch_out_neighbors'] = elapsed / num_runs
    print(f"batch_out_neighbors: {results['batch_out_neighbors']*1000:.4f} ms per run")
    print(f"Speedup: {results['out_neighbors'] / results['batch_out_neighbors']:.2f}x")
    
    test_edges = [(random.randint(0, graph.num_nodes-1), random.randint(0, graph.num_nodes-1)) 
                 for _ in range(100)]
    
    start_time = time.time()
    for _ in range(num_runs):
        for u, v in test_edges:
            graph.get_edge_weight(u, v)
    elapsed = time.time() - start_time
    results['get_edge_weight'] = elapsed / num_runs
    print(f"\nget_edge_weight (individual): {results['get_edge_weight']*1000:.4f} ms per run")
    
    start_time = time.time()
    for _ in range(num_runs):
        graph.batch_get_edge_weight(test_edges)
    elapsed = time.time() - start_time
    results['batch_get_edge_weight'] = elapsed / num_runs
    print(f"batch_get_edge_weight: {results['batch_get_edge_weight']*1000:.4f} ms per run")
    print(f"Speedup: {results['get_edge_weight'] / results['batch_get_edge_weight']:.2f}x")
    
    return results

def benchmark_diffusion_model(model, rounds=1000, num_runs=10):
    print("\n" + "=" * 60)
    print(f"Diffusion Model Benchmark ({model.__class__.__name__})")
    print("=" * 60)
    
    results = {}
    
    print(f"\nRounds: {rounds}")
    print(f"Number of runs: {num_runs}")
    
    start_time = time.time()
    for _ in range(num_runs):
        model.run_monte_carlo_diffusion(rounds=rounds, use_multithread=False)
    elapsed = time.time() - start_time
    results['single_thread'] = elapsed / num_runs
    print(f"Single-thread: {results['single_thread']*1000:.4f} ms per run")
    
    start_time = time.time()
    for _ in range(num_runs):
        model.run_monte_carlo_diffusion(rounds=rounds, use_multithread=True)
    elapsed = time.time() - start_time
    results['multi_thread'] = elapsed / num_runs
    print(f"Multi-thread: {results['multi_thread']*1000:.4f} ms per run")
    print(f"Speedup: {results['single_thread'] / results['multi_thread']:.2f}x")
    
    return results

def run_comprehensive_benchmark():
    print("=" * 60)
    print("PyNetIM Performance Benchmark Suite")
    print("=" * 60)
    
    graph_sizes = [
        (100, 500),
        (1000, 5000),
        (5000, 25000),
    ]
    
    all_results = {}
    
    for num_nodes, num_edges in graph_sizes:
        print(f"\n\n{'='*60}")
        print(f"Testing with {num_nodes} nodes, {num_edges} edges")
        print(f"{'='*60}")
        
        graph = generate_random_graph(num_nodes, num_edges)
        graph_results = benchmark_graph_operations(graph)
        all_results[f'graph_{num_nodes}'] = graph_results
        
        seeds = set(random.sample(range(num_nodes), min(10, num_nodes)))
        
        ic_model = IndependentCascadeModel(graph, seeds)
        ic_results = benchmark_diffusion_model(ic_model, rounds=1000, num_runs=5)
        all_results[f'ic_{num_nodes}'] = ic_results
        
        lt_model = LinearThresholdModel(graph, seeds)
        lt_results = benchmark_diffusion_model(lt_model, rounds=1000, num_runs=5)
        all_results[f'lt_{num_nodes}'] = lt_results
    
    print("\n\n" + "=" * 60)
    print("Summary of Performance Improvements")
    print("=" * 60)
    
    print("\nBatch Operations Speedup:")
    for key in all_results:
        if key.startswith('graph_'):
            results = all_results[key]
            num_nodes = key.split('_')[1]
            print(f"\n{num_nodes} nodes:")
            print(f"  batch_out_degree: {results['out_degree'] / results['batch_out_degree']:.2f}x")
            print(f"  batch_out_neighbors: {results['out_neighbors'] / results['batch_out_neighbors']:.2f}x")
            print(f"  batch_get_edge_weight: {results['get_edge_weight'] / results['batch_get_edge_weight']:.2f}x")
    
    print("\n\nMulti-threading Speedup:")
    for key in all_results:
        if key.startswith('ic_'):
            results = all_results[key]
            num_nodes = key.split('_')[1]
            print(f"\nIC Model ({num_nodes} nodes): {results['single_thread'] / results['multi_thread']:.2f}x")
        
        if key.startswith('lt_'):
            results = all_results[key]
            num_nodes = key.split('_')[1]
            print(f"LT Model ({num_nodes} nodes): {results['single_thread'] / results['multi_thread']:.2f}x")
    
    print("\n" + "=" * 60)
    print("Benchmark completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_comprehensive_benchmark()
