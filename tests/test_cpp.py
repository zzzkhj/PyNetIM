import networkx as nx
from pynetim.cpp.graph import IMGraph
from pynetim.cpp.diffusion_model import IndependentCascadeModel, LinearThresholdModel

g = nx.erdos_renyi_graph(100, 0.2, seed=42, directed=True)
graph = IMGraph(g.number_of_nodes(), g.is_directed())
for u, v in g.edges:
    graph.add_edge(u, v, 1 / g.in_degree(v))
print(graph)

k = 10
# seeds = DegreeDiscountAlgorithm(graph).run(10)
seeds = {37, 30, 93, 24, 73, 10, 20, 27, 28, 45}
print(seeds)

lt = LinearThresholdModel(seeds, graph)
print(lt.run_monte_carlo_diffusion(100, seed=42, use_multithread=True))
#
# si = SusceptibleInfectedModel(graph, seeds, record_states=False)
# # print(si.diffusion(2))
# print(si.run_monte_carlo_diffusion(3, 1, seed=42))

# sir = SusceptibleInfectedRecoveredModel(graph, seeds, gamma=0.5, record_states=False)
# print(sir.run_monte_carlo_diffusion(3, None, seed=42))
