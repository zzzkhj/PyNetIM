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
    edge_weight = [0.5] * len(edges)
    return n, edges, edge_weight


def test_lt_constructor():
    print("\n[测试] LT模型构造函数")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    lt = diffusion_model.LinearThresholdModel(graph, seeds)
    print(f"  基本构造: OK")
    
    lt2 = diffusion_model.LinearThresholdModel(graph, seeds, theta_l=0.0, theta_h=1.0)
    print(f"  带theta参数: OK")
    
    lt3 = diffusion_model.LinearThresholdModel(graph, seeds, theta_l=0.3, theta_h=0.7)
    print(f"  自定义阈值范围[0.3, 0.7]: OK")
    
    lt4 = diffusion_model.LinearThresholdModel(graph, seeds, record_activated=True)
    print(f"  带record_activated参数: OK")
    
    lt5 = diffusion_model.LinearThresholdModel(
        graph, seeds, 
        theta_l=0.2, theta_h=0.8,
        record_activated=True, 
        record_activation_frequency=True
    )
    print(f"  带所有参数: OK")


def test_lt_invalid_theta():
    print("\n[测试] LT模型 无效阈值参数验证")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    try:
        lt = diffusion_model.LinearThresholdModel(graph, seeds, theta_l=-0.1)
        print(f"  theta_l=-0.1: 应该失败但成功了!")
    except ValueError as e:
        print(f"  theta_l=-0.1: 正确抛出异常")
    
    try:
        lt = diffusion_model.LinearThresholdModel(graph, seeds, theta_h=1.5)
        print(f"  theta_h=1.5: 应该失败但成功了!")
    except ValueError as e:
        print(f"  theta_h=1.5: 正确抛出异常")
    
    try:
        lt = diffusion_model.LinearThresholdModel(graph, seeds, theta_l=0.8, theta_h=0.3)
        print(f"  theta_l > theta_h: 应该失败但成功了!")
    except ValueError as e:
        print(f"  theta_l > theta_h: 正确抛出异常")


def test_lt_set_seeds():
    print("\n[测试] LT模型 set_seeds")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    
    lt = diffusion_model.LinearThresholdModel(graph, {0})
    lt.set_seeds({5, 10, 15})
    print(f"  设置新种子集: OK")


def test_lt_record_activated():
    print("\n[测试] LT模型 record_activated 功能")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    lt = diffusion_model.LinearThresholdModel(graph, seeds, record_activated=True)
    
    count = lt.run_single_simulation(seed=42)
    activated = lt.get_activated_nodes()
    print(f"  单次模拟: 激活{count}个节点, 记录了{len(activated)}个节点")
    
    lt.set_record_activated(False)
    count2 = lt.run_single_simulation(seed=42)
    activated2 = lt.get_activated_nodes()
    print(f"  关闭记录后: 激活{count2}个节点, 记录了{len(activated2)}个节点 (应为0)")


def test_lt_activation_frequency():
    print("\n[测试] LT模型 activation_frequency 功能")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    lt = diffusion_model.LinearThresholdModel(graph, seeds, record_activation_frequency=True)
    
    rounds = 100
    avg = lt.run_monte_carlo_diffusion(rounds, seed=42)
    freq = lt.get_activation_frequency()
    total_freq = sum(freq)
    print(f"  蒙特卡洛模拟({rounds}轮): 平均激活{avg:.1f}个节点")
    print(f"  激活频率总和: {total_freq} (应约等于{avg * rounds:.0f})")
    
    lt.set_record_activation_frequency(False)
    freq2 = lt.get_activation_frequency()
    print(f"  关闭记录后频率总和: {sum(freq2)} (应为0)")


def test_lt_single_simulation():
    print("\n[测试] LT模型 run_single_simulation")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    lt = diffusion_model.LinearThresholdModel(graph, seeds)
    
    count1 = lt.run_single_simulation(seed=42)
    count2 = lt.run_single_simulation(seed=42)
    print(f"  相同种子重复性: count1={count1}, count2={count2} (应相同)")
    
    count3 = lt.run_single_simulation(seed=100)
    count4 = lt.run_single_simulation(seed=200)
    print(f"  不同种子: seed100={count3}, seed200={count4}")
    
    count5 = lt.run_single_simulation()
    print(f"  随机种子: count={count5}")


def test_lt_monte_carlo():
    print("\n[测试] LT模型 run_monte_carlo_diffusion")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    lt = diffusion_model.LinearThresholdModel(graph, seeds)
    
    avg1 = lt.run_monte_carlo_diffusion(100, seed=42)
    avg2 = lt.run_monte_carlo_diffusion(100, seed=42)
    print(f"  相同种子重复性: avg1={avg1:.2f}, avg2={avg2:.2f} (应相同)")
    
    avg3 = lt.run_monte_carlo_diffusion(1000, seed=42)
    print(f"  1000轮模拟: 平均激活{avg3:.2f}个节点")
    
    avg4 = lt.run_monte_carlo_diffusion(100, seed=42, use_multithread=True)
    print(f"  多线程模式: 平均激活{avg4:.2f}个节点")
    
    avg5 = lt.run_monte_carlo_diffusion(100, seed=42, use_multithread=True, num_threads=2)
    print(f"  指定2线程: 平均激活{avg5:.2f}个节点")


def test_lt_threshold_range():
    print("\n[测试] LT模型 阈值范围影响")
    n = 20
    edges = [[i, i+1] for i in range(n-1)]
    edge_weight = [1.0] * len(edges)
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    lt_low = diffusion_model.LinearThresholdModel(graph, seeds, theta_l=0.0, theta_h=0.1)
    avg_low = lt_low.run_monte_carlo_diffusion(1000, seed=42)
    
    lt_high = diffusion_model.LinearThresholdModel(graph, seeds, theta_l=0.9, theta_h=1.0)
    avg_high = lt_high.run_monte_carlo_diffusion(1000, seed=42)
    
    lt_mid = diffusion_model.LinearThresholdModel(graph, seeds, theta_l=0.4, theta_h=0.6)
    avg_mid = lt_mid.run_monte_carlo_diffusion(1000, seed=42)
    
    print(f"  低阈值[0.0, 0.1]: 平均激活{avg_low:.2f}个节点 (应较高)")
    print(f"  中阈值[0.4, 0.6]: 平均激活{avg_mid:.2f}个节点")
    print(f"  高阈值[0.9, 1.0]: 平均激活{avg_high:.2f}个节点 (应较低)")
    print(f"  低阈值 > 中阈值 > 高阈值: {avg_low > avg_mid > avg_high}")


def test_lt_monte_carlo_with_activated_nodes():
    print("\n[测试] LT模型 蒙特卡洛+激活节点记录")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    lt = diffusion_model.LinearThresholdModel(graph, seeds, record_activated=True)
    
    avg = lt.run_monte_carlo_diffusion(100, seed=42)
    activated = lt.get_activated_nodes()
    print(f"  100轮蒙特卡洛: 平均激活{avg:.2f}个节点")
    print(f"  激活节点并集: {len(activated)}个节点")


def test_lt_monte_carlo_with_frequency():
    print("\n[测试] LT模型 蒙特卡洛+激活频率记录")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    lt = diffusion_model.LinearThresholdModel(graph, seeds, record_activation_frequency=True)
    
    rounds = 100
    avg = lt.run_monte_carlo_diffusion(rounds, seed=42)
    freq = lt.get_activation_frequency()
    
    seed_freq = freq[0]
    print(f"  种子节点0被激活次数: {seed_freq} (应为{rounds})")
    
    most_activated = max(freq)
    most_activated_node = freq.index(most_activated)
    print(f"  最常被激活的节点: {most_activated_node} (激活{most_activated}次)")


def test_lt_edge_weights():
    print("\n[测试] LT模型 边权值影响")
    n = 10
    edges = [[i, i+1] for i in range(n-1)]
    
    low_weight = [0.1] * len(edges)
    graph_low = im_graph.IMGraphCpp(n, edges, low_weight, True)
    lt_low = diffusion_model.LinearThresholdModel(graph_low, {0})
    avg_low = lt_low.run_monte_carlo_diffusion(1000, seed=42)
    
    high_weight = [1.0] * len(edges)
    graph_high = im_graph.IMGraphCpp(n, edges, high_weight, True)
    lt_high = diffusion_model.LinearThresholdModel(graph_high, {0})
    avg_high = lt_high.run_monte_carlo_diffusion(1000, seed=42)
    
    print(f"  低权值(0.1): 平均激活{avg_low:.2f}个节点")
    print(f"  高权值(1.0): 平均激活{avg_high:.2f}个节点")
    print(f"  高权值应大于低权值: {avg_high > avg_low}")


def run_all_tests():
    print("=" * 60)
    print("LT (Linear Threshold) 模型完整测试")
    print("=" * 60)
    
    test_lt_constructor()
    test_lt_invalid_theta()
    test_lt_set_seeds()
    test_lt_record_activated()
    test_lt_activation_frequency()
    test_lt_single_simulation()
    test_lt_monte_carlo()
    test_lt_threshold_range()
    test_lt_monte_carlo_with_activated_nodes()
    test_lt_monte_carlo_with_frequency()
    test_lt_edge_weights()
    
    print("\n" + "=" * 60)
    print("所有LT模型测试通过!")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
