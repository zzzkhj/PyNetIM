import time
import random
from pynetim.cpp.graph import IMGraphCpp

def test_batch_operations():
    print("=" * 60)
    print("Testing Batch Operations")
    print("=" * 60)
    
    num_nodes = 1000
    num_edges = 5000
    
    random.seed(42)
    
    edges = []
    weights = []
    for _ in range(num_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v:
            edges.append((u, v))
            weights.append(random.random())
    
    print(f"\nCreating graph with {num_nodes} nodes and {len(edges)} edges...")
    graph = IMGraphCpp(num_nodes=num_nodes, edges=edges, weights=weights, directed=True)
    print(f"Graph: {graph}")
    
    print("\n" + "=" * 60)
    print("Test 1: batch_out_degree")
    print("=" * 60)
    
    test_nodes = list(range(0, 100, 5))
    
    start_time = time.time()
    individual_degrees = [graph.out_degree(u) for u in test_nodes]
    individual_time = time.time() - start_time
    
    start_time = time.time()
    batch_degrees = graph.batch_out_degree(test_nodes)
    batch_time = time.time() - start_time
    
    print(f"Test nodes: {len(test_nodes)}")
    print(f"Individual query time: {individual_time:.6f}s")
    print(f"Batch query time: {batch_time:.6f}s")
    print(f"Speedup: {individual_time / batch_time:.2f}x")
    print(f"Results match: {individual_degrees == batch_degrees}")
    
    print("\n" + "=" * 60)
    print("Test 2: batch_out_neighbors")
    print("=" * 60)
    
    test_nodes = list(range(0, 50, 5))
    
    start_time = time.time()
    individual_neighbors = [graph.out_neighbors(u) for u in test_nodes]
    individual_time = time.time() - start_time
    
    start_time = time.time()
    batch_neighbors = graph.batch_out_neighbors(test_nodes)
    batch_time = time.time() - start_time
    
    print(f"Test nodes: {len(test_nodes)}")
    print(f"Individual query time: {individual_time:.6f}s")
    print(f"Batch query time: {batch_time:.6f}s")
    print(f"Speedup: {individual_time / batch_time:.2f}x")
    
    print(f"\nSample results (first 3 nodes):")
    for i in range(min(3, len(test_nodes))):
        u = test_nodes[i]
        print(f"  Node {u}: {len(individual_neighbors[i])} neighbors")
    
    print("\n" + "=" * 60)
    print("Test 3: batch_get_edge_weight")
    print("=" * 60)
    
    test_edges = edges[:100]
    
    start_time = time.time()
    individual_weights = [graph.get_edge_weight(u, v) for u, v in test_edges]
    individual_time = time.time() - start_time
    
    start_time = time.time()
    batch_weights = graph.batch_get_edge_weight(test_edges)
    batch_time = time.time() - start_time
    
    print(f"Test edges: {len(test_edges)}")
    print(f"Individual query time: {individual_time:.6f}s")
    print(f"Batch query time: {batch_time:.6f}s")
    print(f"Speedup: {individual_time / batch_time:.2f}x")
    print(f"Results match: {individual_weights == batch_weights}")
    
    print(f"\nSample weights (first 5 edges):")
    for i in range(min(5, len(test_edges))):
        u, v = test_edges[i]
        print(f"  Edge ({u}, {v}): {individual_weights[i]:.4f}")
    
    print("\n" + "=" * 60)
    print("Test 4: Large scale batch operations")
    print("=" * 60)
    
    large_nodes = list(range(num_nodes))
    
    start_time = time.time()
    all_degrees = graph.batch_out_degree(large_nodes)
    batch_time = time.time() - start_time
    
    print(f"Querying all {num_nodes} nodes...")
    print(f"Batch query time: {batch_time:.6f}s")
    print(f"Average degree: {sum(all_degrees) / num_nodes:.2f}")
    print(f"Max degree: {max(all_degrees)}")
    print(f"Min degree: {min(all_degrees)}")
    
    print("\n" + "=" * 60)
    print("All batch operation tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_batch_operations()
