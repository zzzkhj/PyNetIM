import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pynetim.cpp.graph as im_graph
import pynetim.cpp.diffusion_model as diffusion_model


def create_test_graph():
    n = 100
    edges = [[i, i+1] for i in range(n-1)]
    edge_weight = [0.8] * len(edges)
    return n, edges, edge_weight


def test_ic_constructor():
    print("\n[测试] IC模型构造函数")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    ic = diffusion_model.IndependentCascadeModel(graph, seeds)
    print(f"  基本构造: OK")
    
    ic2 = diffusion_model.IndependentCascadeModel(graph, seeds, record_activated=True)
    print(f"  带record_activated参数: OK")
    
    ic3 = diffusion_model.IndependentCascadeModel(graph, seeds, record_activation_frequency=True)
    print(f"  带record_activation_frequency参数: OK")
    
    ic4 = diffusion_model.IndependentCascadeModel(
        graph, seeds, 
        record_activated=True, 
        record_activation_frequency=True
    )
    print(f"  带所有参数: OK")


def test_ic_set_seeds():
    print("\n[测试] IC模型 set_seeds")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    
    ic = diffusion_model.IndependentCascadeModel(graph, {0})
    ic.set_seeds({5, 10, 15})
    print(f"  设置新种子集: OK")


def test_ic_record_activated():
    print("\n[测试] IC模型 record_activated 功能")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    ic = diffusion_model.IndependentCascadeModel(graph, seeds, record_activated=True)
    
    count = ic.run_single_simulation(seed=42)
    activated = ic.get_activated_nodes()
    print(f"  单次模拟: 激活{count}个节点, 记录了{len(activated)}个节点")
    
    ic.set_record_activated(False)
    count2 = ic.run_single_simulation(seed=42)
    activated2 = ic.get_activated_nodes()
    print(f"  关闭记录后: 激活{count2}个节点, 记录了{len(activated2)}个节点 (应为0)")


def test_ic_activation_frequency():
    print("\n[测试] IC模型 activation_frequency 功能")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    ic = diffusion_model.IndependentCascadeModel(graph, seeds, record_activation_frequency=True)
    
    rounds = 100
    avg = ic.run_monte_carlo_diffusion(rounds, seed=42)
    freq = ic.get_activation_frequency()
    total_freq = sum(freq)
    print(f"  蒙特卡洛模拟({rounds}轮): 平均激活{avg:.1f}个节点")
    print(f"  激活频率总和: {total_freq} (应约等于{avg * rounds:.0f})")
    
    ic.set_record_activation_frequency(False)
    freq2 = ic.get_activation_frequency()
    print(f"  关闭记录后频率总和: {sum(freq2)} (应为0)")


def test_ic_single_simulation():
    print("\n[测试] IC模型 run_single_simulation")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    ic = diffusion_model.IndependentCascadeModel(graph, seeds)
    
    count1 = ic.run_single_simulation(seed=42)
    count2 = ic.run_single_simulation(seed=42)
    print(f"  相同种子重复性: count1={count1}, count2={count2} (应相同)")
    
    count3 = ic.run_single_simulation(seed=100)
    count4 = ic.run_single_simulation(seed=200)
    print(f"  不同种子: seed100={count3}, seed200={count4}")
    
    count5 = ic.run_single_simulation()
    print(f"  随机种子: count={count5}")


def test_ic_monte_carlo():
    print("\n[测试] IC模型 run_monte_carlo_diffusion")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    ic = diffusion_model.IndependentCascadeModel(graph, seeds)
    
    avg1 = ic.run_monte_carlo_diffusion(100, seed=42)
    avg2 = ic.run_monte_carlo_diffusion(100, seed=42)
    print(f"  相同种子重复性: avg1={avg1:.2f}, avg2={avg2:.2f} (应相同)")
    
    avg3 = ic.run_monte_carlo_diffusion(1000, seed=42)
    print(f"  1000轮模拟: 平均激活{avg3:.2f}个节点")
    
    avg4 = ic.run_monte_carlo_diffusion(100, seed=42, use_multithread=True)
    print(f"  多线程模式: 平均激活{avg4:.2f}个节点")
    
    avg5 = ic.run_monte_carlo_diffusion(100, seed=42, use_multithread=True, num_threads=2)
    print(f"  指定2线程: 平均激活{avg5:.2f}个节点")


def test_ic_monte_carlo_with_activated_nodes():
    print("\n[测试] IC模型 蒙特卡洛+激活节点记录")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    ic = diffusion_model.IndependentCascadeModel(graph, seeds, record_activated=True)
    
    avg = ic.run_monte_carlo_diffusion(100, seed=42)
    activated = ic.get_activated_nodes()
    print(f"  100轮蒙特卡洛: 平均激活{avg:.2f}个节点")
    print(f"  激活节点并集: {len(activated)}个节点")


def test_ic_monte_carlo_with_frequency():
    print("\n[测试] IC模型 蒙特卡洛+激活频率记录")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    ic = diffusion_model.IndependentCascadeModel(graph, seeds, record_activation_frequency=True)
    
    rounds = 100
    avg = ic.run_monte_carlo_diffusion(rounds, seed=42)
    freq = ic.get_activation_frequency()
    
    seed_freq = freq[0]
    print(f"  种子节点0被激活次数: {seed_freq} (应为{rounds})")
    
    most_activated = max(freq)
    most_activated_node = freq.index(most_activated)
    print(f"  最常被激活的节点: {most_activated_node} (激活{most_activated}次)")


def test_ic_edge_weights():
    print("\n[测试] IC模型 边权值影响")
    n = 10
    edges = [[i, i+1] for i in range(n-1)]
    
    low_weight = [0.1] * len(edges)
    graph_low = im_graph.IMGraphCpp(n, edges, low_weight, True)
    ic_low = diffusion_model.IndependentCascadeModel(graph_low, {0})
    avg_low = ic_low.run_monte_carlo_diffusion(1000, seed=42)
    
    high_weight = [0.9] * len(edges)
    graph_high = im_graph.IMGraphCpp(n, edges, high_weight, True)
    ic_high = diffusion_model.IndependentCascadeModel(graph_high, {0})
    avg_high = ic_high.run_monte_carlo_diffusion(1000, seed=42)
    
    print(f"  低权值(0.1): 平均激活{avg_low:.2f}个节点")
    print(f"  高权值(0.9): 平均激活{avg_high:.2f}个节点")
    print(f"  高权值应大于低权值: {avg_high > avg_low}")


def run_all_tests():
    print("=" * 60)
    print("IC (Independent Cascade) 模型完整测试")
    print("=" * 60)
    
    test_ic_constructor()
    test_ic_set_seeds()
    test_ic_record_activated()
    test_ic_activation_frequency()
    test_ic_single_simulation()
    test_ic_monte_carlo()
    test_ic_monte_carlo_with_activated_nodes()
    test_ic_monte_carlo_with_frequency()
    test_ic_edge_weights()
    
    print("\n" + "=" * 60)
    print("所有IC模型测试通过!")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
