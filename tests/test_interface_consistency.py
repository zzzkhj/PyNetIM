import random
import networkx as nx
from pynetim.py.graph import IMGraph as PyIMGraph
from pynetim.cpp.graph import IMGraphCpp

def test_interface_consistency():
    print("=" * 60)
    print("Testing Python/C++ Interface Consistency")
    print("=" * 60)
    
    num_nodes = 100
    num_edges = 500
    
    random.seed(42)
    
    nx_graph = nx.DiGraph()
    for _ in range(num_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v:
            weight = random.random()
            nx_graph.add_edge(u, v, weight=weight)
    
    py_graph = PyIMGraph(nx_graph, edge_weight_type='WC')
    
    edges_list = [(u, v) for u, v in nx_graph.edges()]
    weights_list = [nx_graph[u][v]['weight'] for u, v in nx_graph.edges()]
    
    cpp_graph = IMGraphCpp(num_nodes=num_nodes, edges=edges_list, weights=weights_list, directed=True)
    
    test_nodes = list(range(0, 50, 5))
    
    print("\n" + "=" * 60)
    print("Test 1: batch_out_degree")
    print("=" * 60)
    py_out_degrees = py_graph.batch_out_degree(test_nodes)
    cpp_out_degrees = cpp_graph.batch_out_degree(test_nodes)
    
    print(f"Python result: {py_out_degrees[:5]}")
    print(f"C++ result: {cpp_out_degrees[:5]}")
    print(f"Results match: {py_out_degrees == cpp_out_degrees}")
    
    print("\n" + "=" * 60)
    print("Test 2: batch_in_degree")
    print("=" * 60)
    py_in_degrees = py_graph.batch_in_degree(test_nodes)
    cpp_in_degrees = cpp_graph.batch_in_degree(test_nodes)
    
    print(f"Python result: {py_in_degrees[:5]}")
    print(f"C++ result: {cpp_in_degrees[:5]}")
    print(f"Results match: {py_in_degrees == cpp_in_degrees}")
    
    print("\n" + "=" * 60)
    print("Test 3: batch_degree")
    print("=" * 60)
    py_degrees = py_graph.batch_degree(test_nodes)
    cpp_degrees = cpp_graph.batch_degree(test_nodes)
    
    print(f"Python result: {py_degrees[:5]}")
    print(f"C++ result: {cpp_degrees[:5]}")
    print(f"Results match: {py_degrees == cpp_degrees}")
    
    print("\n" + "=" * 60)
    print("Test 4: batch_out_neighbors")
    print("=" * 60)
    py_neighbors = py_graph.batch_out_degree(test_nodes[:5])
    cpp_neighbors = cpp_graph.batch_out_neighbors(test_nodes[:5])
    
    print(f"Python degrees: {py_neighbors}")
    print(f"C++ neighbor counts: {[len(n) for n in cpp_neighbors]}")
    
    print("\n" + "=" * 60)
    print("Test 5: batch_get_edge_weight")
    print("=" * 60)
    test_edges = [(test_nodes[i], test_nodes[(i+1) % len(test_nodes)]) for i in range(10)]
    
    py_weights = []
    cpp_weights = cpp_graph.batch_get_edge_weight(test_edges)
    
    for u, v in test_edges:
        if nx_graph.has_edge(u, v):
            py_weights.append(nx_graph[u][v]['weight'])
        else:
            py_weights.append(0.0)
    
    print(f"Python result: {py_weights[:5]}")
    print(f"C++ result: {cpp_weights[:5]}")
    print(f"Results match: {py_weights == cpp_weights}")
    
    print("\n" + "=" * 60)
    print("Interface consistency tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_interface_consistency()
