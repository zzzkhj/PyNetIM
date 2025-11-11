使用示例
========

完整示例：社交网络影响力分析
----------------------------

.. code-block:: python

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
   influence_spread = IndependentCascadeModel(G, seeds, p=0.1, mc=1000).run_monte_carlo_diffusion(1000, seed=42)

   print(f"种子节点: {seeds}")
   print(f"平均影响节点数: {influence_spread}")

更多示例
--------

暂无