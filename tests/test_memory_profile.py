import random
from pynetim.cpp.graph import IMGraphCpp
from pynetim.cpp.diffusion_model import IndependentCascadeModel, LinearThresholdModel
from memory_monitor import MemoryMonitor, profile_memory_usage, print_memory_stats, format_memory_size

def generate_random_graph(num_nodes, num_edges, directed=True, seed=42):
    random.seed(seed)
    
    edges = []
    weights = []
    
    for _ in range(num_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v:
            edges.append((u, v))
            weights.append(random.random())
    
    return IMGraphCpp(num_nodes=num_nodes, edges=edges, weights=weights, directed=directed)

def profile_graph_creation():
    print("\n" + "=" * 60)
    print("Profiling Graph Creation")
    print("=" * 60)
    
    sizes = [
        (100, 500),
        (1000, 5000),
        (5000, 25000),
        (10000, 50000),
    ]
    
    for num_nodes, num_edges in sizes:
        print(f"\nCreating graph with {num_nodes} nodes and {num_edges} edges...")
        stats = profile_memory_usage(generate_random_graph, num_nodes, num_edges)
        print_memory_stats(stats, f"Graph ({num_nodes} nodes, {num_edges} edges)")

def profile_graph_operations():
    print("\n" + "=" * 60)
    print("Profiling Graph Operations")
    print("=" * 60)
    
    graph = generate_random_graph(10000, 50000)
    print(f"Graph: {graph}")
    print(f"Baseline memory: {format_memory_size(MemoryMonitor().get_memory_usage())}")
    
    monitor = MemoryMonitor()
    
    test_nodes = list(range(0, 10000, 100))
    
    print(f"\nTesting batch_out_degree with {len(test_nodes)} nodes...")
    monitor.reset_baseline()
    degrees = graph.batch_out_degree(test_nodes)
    print(f"Memory increase: {format_memory_size(monitor.get_memory_increase())}")
    print(f"Peak memory: {format_memory_size(monitor.get_peak_memory())}")
    
    print(f"\nTesting batch_out_neighbors with {len(test_nodes)} nodes...")
    monitor.reset_baseline()
    neighbors = graph.batch_out_neighbors(test_nodes)
    print(f"Memory increase: {format_memory_size(monitor.get_memory_increase())}")
    print(f"Peak memory: {format_memory_size(monitor.get_peak_memory())}")
    
    test_edges = [(random.randint(0, 9999), random.randint(0, 9999)) for _ in range(1000)]
    
    print(f"\nTesting batch_get_edge_weight with {len(test_edges)} edges...")
    monitor.reset_baseline()
    weights = graph.batch_get_edge_weight(test_edges)
    print(f"Memory increase: {format_memory_size(monitor.get_memory_increase())}")
    print(f"Peak memory: {format_memory_size(monitor.get_peak_memory())}")

def profile_diffusion_models():
    print("\n" + "=" * 60)
    print("Profiling Diffusion Models")
    print("=" * 60)
    
    graph = generate_random_graph(10000, 50000)
    seeds = set(random.sample(range(10000), 50))
    
    print(f"\nGraph: {graph}")
    print(f"Seeds: {len(seeds)} nodes")
    print(f"Baseline memory: {format_memory_size(MemoryMonitor().get_memory_usage())}")
    
    monitor = MemoryMonitor()
    
    print(f"\nCreating IndependentCascadeModel...")
    monitor.reset_baseline()
    ic_model = IndependentCascadeModel(graph, seeds)
    print(f"Memory increase: {format_memory_size(monitor.get_memory_increase())}")
    print(f"Peak memory: {format_memory_size(monitor.get_peak_memory())}")
    
    print(f"\nRunning IC model (1000 rounds, single-thread)...")
    monitor.reset_baseline()
    result = ic_model.run_monte_carlo_diffusion(rounds=1000, use_multithread=False)
    print(f"Result: {result:.4f}")
    print(f"Memory increase: {format_memory_size(monitor.get_memory_increase())}")
    print(f"Peak memory: {format_memory_size(monitor.get_peak_memory())}")
    
    print(f"\nRunning IC model (1000 rounds, multi-thread)...")
    monitor.reset_baseline()
    result = ic_model.run_monte_carlo_diffusion(rounds=1000, use_multithread=True)
    print(f"Result: {result:.4f}")
    print(f"Memory increase: {format_memory_size(monitor.get_memory_increase())}")
    print(f"Peak memory: {format_memory_size(monitor.get_peak_memory())}")
    
    print(f"\nCreating LinearThresholdModel...")
    monitor.reset_baseline()
    lt_model = LinearThresholdModel(graph, seeds)
    print(f"Memory increase: {format_memory_size(monitor.get_memory_increase())}")
    print(f"Peak memory: {format_memory_size(monitor.get_peak_memory())}")
    
    print(f"\nRunning LT model (1000 rounds, single-thread)...")
    monitor.reset_baseline()
    result = lt_model.run_monte_carlo_diffusion(rounds=1000, use_multithread=False)
    print(f"Result: {result:.4f}")
    print(f"Memory increase: {format_memory_size(monitor.get_memory_increase())}")
    print(f"Peak memory: {format_memory_size(monitor.get_peak_memory())}")
    
    print(f"\nRunning LT model (1000 rounds, multi-thread)...")
    monitor.reset_baseline()
    result = lt_model.run_monte_carlo_diffusion(rounds=1000, use_multithread=True)
    print(f"Result: {result:.4f}")
    print(f"Memory increase: {format_memory_size(monitor.get_memory_increase())}")
    print(f"Peak memory: {format_memory_size(monitor.get_peak_memory())}")

def profile_memory_efficiency():
    print("\n" + "=" * 60)
    print("Memory Efficiency Analysis")
    print("=" * 60)
    
    sizes = [
        (1000, 5000),
        (5000, 25000),
        (10000, 50000),
    ]
    
    print("\nGraph Size vs Memory Usage:")
    print(f"{'Nodes':<10} {'Edges':<10} {'Memory (MB)':<15} {'Memory/Node (KB)':<20} {'Memory/Edge (KB)':<20}")
    print("-" * 75)
    
    for num_nodes, num_edges in sizes:
        stats = profile_memory_usage(generate_random_graph, num_nodes, num_edges)
        memory_mb = stats['memory_increase_mb']
        memory_per_node_kb = (memory_mb * 1024) / num_nodes
        memory_per_edge_kb = (memory_mb * 1024) / num_edges
        
        print(f"{num_nodes:<10} {num_edges:<10} {memory_mb:<15.2f} {memory_per_node_kb:<20.2f} {memory_per_edge_kb:<20.2f}")

def run_comprehensive_memory_profile():
    print("=" * 60)
    print("PyNetIM Memory Profiling Suite")
    print("=" * 60)
    
    profile_graph_creation()
    profile_graph_operations()
    profile_diffusion_models()
    profile_memory_efficiency()
    
    print("\n" + "=" * 60)
    print("Memory profiling completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_comprehensive_memory_profile()
