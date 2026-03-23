import random
from pynetim.cpp.graph import IMGraphCpp

def test_unified_batch_interfaces():
    print("=" * 60)
    print("Testing Unified Batch Interfaces")
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
    
    test_nodes = list(range(0, 100, 5))
    
    print("\n" + "=" * 60)
    print("Test 1: batch_out_degree")
    print("=" * 60)
    out_degrees = graph.batch_out_degree(test_nodes)
    print(f"Test nodes: {len(test_nodes)}")
    print(f"Sample results (first 5):")
    for i in range(min(5, len(test_nodes))):
        print(f"  Node {test_nodes[i]}: out_degree = {out_degrees[i]}")
    
    print("\n" + "=" * 60)
    print("Test 2: batch_in_degree")
    print("=" * 60)
    in_degrees = graph.batch_in_degree(test_nodes)
    print(f"Test nodes: {len(test_nodes)}")
    print(f"Sample results (first 5):")
    for i in range(min(5, len(test_nodes))):
        print(f"  Node {test_nodes[i]}: in_degree = {in_degrees[i]}")
    
    print("\n" + "=" * 60)
    print("Test 3: batch_degree")
    print("=" * 60)
    degrees = graph.batch_degree(test_nodes)
    print(f"Test nodes: {len(test_nodes)}")
    print(f"Sample results (first 5):")
    for i in range(min(5, len(test_nodes))):
        print(f"  Node {test_nodes[i]}: degree = {degrees[i]}")
        print(f"    (out_degree={out_degrees[i]}, in_degree={in_degrees[i]}, total={degrees[i]})")
    
    print("\n" + "=" * 60)
    print("Test 4: batch_out_neighbors")
    print("=" * 60)
    neighbors = graph.batch_out_neighbors(test_nodes[:5])
    print(f"Test nodes: 5")
    for i, node in enumerate(test_nodes[:5]):
        print(f"  Node {node}: {len(neighbors[i])} neighbors")
        if neighbors[i]:
            print(f"    First neighbor: ({neighbors[i][0][0]}, weight={neighbors[i][0][1]:.4f})")
    
    print("\n" + "=" * 60)
    print("Test 5: batch_get_edge_weight")
    print("=" * 60)
    test_edges = [(test_nodes[i], test_nodes[(i+1) % len(test_nodes)]) for i in range(10)]
    edge_weights = graph.batch_get_edge_weight(test_edges)
    print(f"Test edges: {len(test_edges)}")
    print(f"Sample results:")
    for i, (u, v) in enumerate(test_edges[:5]):
        print(f"  Edge ({u}, {v}): weight = {edge_weights[i]:.4f}")
    
    print("\n" + "=" * 60)
    print("All unified batch interface tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_unified_batch_interfaces()
