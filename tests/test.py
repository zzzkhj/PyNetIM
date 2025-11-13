import networkx as nx
from pynetim.graph import IMGraph
from pynetim.algorithms import DegreeDiscountAlgorithm
from pynetim.diffusion_model import IndependentCascadeModel, SusceptibleInfectedModel, SusceptibleInfectedRecoveredModel

g = nx.erdos_renyi_graph(100, 0.2, seed=42)
graph = IMGraph(g, edge_weight_type='WC')
print(graph)

k = 10
seeds = DegreeDiscountAlgorithm(graph).run(10)
print(seeds)
#
# si = SusceptibleInfectedModel(graph, seeds, record_states=False)
# # print(si.diffusion(2))
# print(si.run_monte_carlo_diffusion(3, 1, seed=42))

sir = SusceptibleInfectedRecoveredModel(graph, seeds, gamma=0.5, record_states=False)
print(sir.run_monte_carlo_diffusion(3, None, seed=42))
