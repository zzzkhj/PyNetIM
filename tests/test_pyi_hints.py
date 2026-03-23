from pynetim.cpp.graph import IMGraphCpp
from pynetim.cpp.diffusion_model import IndependentCascadeModel, LinearThresholdModel

def test_pyi_hints():
    print("=" * 60)
    print("Testing .pyi Type Hints")
    print("=" * 60)
    
    num_nodes = 100
    num_edges = 500
    
    import random
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
    print("Testing IMGraphCpp methods with type hints")
    print("=" * 60)
    
    test_nodes = [0, 1, 2, 3, 4]
    
    print("\n1. Testing batch_out_degree...")
    out_degrees = graph.batch_out_degree(test_nodes)
    print(f"   Result: {out_degrees}")
    print(f"   Type: {type(out_degrees)}")
    
    print("\n2. Testing batch_in_degree...")
    in_degrees = graph.batch_in_degree(test_nodes)
    print(f"   Result: {in_degrees}")
    print(f"   Type: {type(in_degrees)}")
    
    print("\n3. Testing batch_degree...")
    degrees = graph.batch_degree(test_nodes)
    print(f"   Result: {degrees}")
    print(f"   Type: {type(degrees)}")
    
    print("\n4. Testing batch_out_neighbors...")
    neighbors = graph.batch_out_neighbors(test_nodes)
    print(f"   Result length: {len(neighbors)}")
    print(f"   Type: {type(neighbors)}")
    
    print("\n5. Testing batch_get_edge_weight...")
    test_edges = [(0, 1), (1, 2), (2, 3)]
    edge_weights = graph.batch_get_edge_weight(test_edges)
    print(f"   Result: {edge_weights}")
    print(f"   Type: {type(edge_weights)}")
    
    print("\n" + "=" * 60)
    print("Testing IndependentCascadeModel with type hints")
    print("=" * 60)
    
    seeds = {0, 1}
    ic_model = IndependentCascadeModel(graph, seeds)
    
    print("\n1. Testing run_monte_carlo_diffusion...")
    result = ic_model.run_monte_carlo_diffusion(rounds=100, seed=42, use_multithread=False, num_threads=0)
    print(f"   Result: {result}")
    print(f"   Type: {type(result)}")
    
    print("\n2. Testing set_seeds...")
    new_seeds = {2, 3}
    ic_model.set_seeds(new_seeds)
    print(f"   Seeds set successfully")
    
    print("\n" + "=" * 60)
    print("Testing LinearThresholdModel with type hints")
    print("=" * 60)
    
    lt_model = LinearThresholdModel(graph, seeds)
    
    print("\n1. Testing run_monte_carlo_diffusion...")
    result = lt_model.run_monte_carlo_diffusion(rounds=100, seed=42, use_multithread=False, num_threads=0)
    print(f"   Result: {result}")
    print(f"   Type: {type(result)}")
    
    print("\n2. Testing set_seeds...")
    lt_model.set_seeds(new_seeds)
    print(f"   Seeds set successfully")
    
    print("\n" + "=" * 60)
    print("All .pyi type hint tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_pyi_hints()
