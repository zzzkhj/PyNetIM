import networkx as nx
from pynetim.graph import IMGraph
from pynetim.algorithms import DegreeDiscountAlgorithm
from pynetim.diffusion_model import IndependentCascadeModel

g = nx.erdos_renyi_graph(100, 0.2, seed=42)
graph = IMGraph(g, edge_weight_type='WC')
print(graph)

k = 10
dd = DegreeDiscountAlgorithm(graph)
seeds = dd.run(k)
print(seeds)

ic = IndependentCascadeModel(graph, seeds, record_states=False)
influence_spread = ic.run_monte_carlo_diffusion(1000, seed=42)
print(influence_spread)



import networkx as nx
from pynetim.graph import IMGraph
from pynetim.algorithms import CELFAlgorithm
from pynetim.diffusion_model import IndependentCascadeModel

# 创建社交网络图
G = nx.karate_club_graph()
graph = IMGraph(G, edge_weight_type='WC')

# 运行贪心算法选择种子
k = 5
seeds = CELFAlgorithm(graph, IndependentCascadeModel).run(k=k, round=1000, seed=42)

# 评估影响力扩散
influence_spread = IndependentCascadeModel(G, seeds).run_monte_carlo_diffusion(1000, seed=42)

print(f"种子节点: {seeds}")
print(f"平均影响节点数: {influence_spread}")