import networkx as nx

import pynetim as im

g = nx.erdos_renyi_graph(100, 0.2, seed=42)
graph = im.IMGraph(g, edge_weight_type='WC')
print(graph)

k = 10
dd = im.algorithms.DegreeDiscountAlgorithm(graph)
seeds = dd.run(k)
print(seeds)

ic = im.IndependentCascadeModel(graph, seeds, record_states=False)
# influence_spread = ic.run_monte_carlo_diffusion(1000, seed=42)
influence_spread = im.run_monte_carlo_diffusion(ic, 1000, seed=42)
print(influence_spread)


