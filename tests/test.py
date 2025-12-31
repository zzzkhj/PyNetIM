# import networkx as nx
# # from pynetim.py.graph import IMGraph
# # from pynetim.py.algorithms import DegreeDiscountAlgorithm
# # from pynetim.py.diffusion_model import IndependentCascadeModel, SusceptibleInfectedModel, SusceptibleInfectedRecoveredModel,\
# #     LinearThresholdModel
#
# g = nx.erdos_renyi_graph(100, 0.2, seed=42)
# # print(g.edges)
# print(g.edges[(0, 2)])
#
# graph = IMGraph(g, edge_weight_type='WC')
# print(graph)
#
# k = 10
# seeds = DegreeDiscountAlgorithm(graph).run(10)
# print(seeds)
#
# lt = LinearThresholdModel(graph, seeds, record_states=False)
# print(lt.run_monte_carlo_diffusion(100, None, seed=42))
# #
# # si = SusceptibleInfectedModel(graph, seeds, record_states=False)
# # # print(si.diffusion(2))
# # print(si.run_monte_carlo_diffusion(3, 1, seed=42))
#
# # sir = SusceptibleInfectedRecoveredModel(graph, seeds, gamma=0.5, record_states=False)
# # print(sir.run_monte_carlo_diffusion(3, None, seed=42))
