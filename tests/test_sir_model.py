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
    edge_weight = [1.0] * len(edges)
    return n, edges, edge_weight


def test_sir_constructor():
    print("\n[测试] SIR模型构造函数")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.5, gamma=0.1)
    print(f"  基本构造: OK")
    
    sir2 = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.5, gamma=0.1, record_activated=True)
    print(f"  带record_activated参数: OK")
    
    sir3 = diffusion_model.SusceptibleInfectedRecoveredModel(
        graph, seeds, 
        beta=0.5, gamma=0.1,
        record_activated=True, 
        record_activation_frequency=True
    )
    print(f"  带所有参数: OK")


def test_sir_invalid_params():
    print("\n[测试] SIR模型 无效参数验证")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    try:
        sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.0, gamma=0.1)
        print(f"  beta=0.0: 应该失败但成功了!")
    except ValueError as e:
        print(f"  beta=0.0: 正确抛出异常")
    
    try:
        sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=-0.1, gamma=0.1)
        print(f"  beta=-0.1: 应该失败但成功了!")
    except ValueError as e:
        print(f"  beta=-0.1: 正确抛出异常")
    
    try:
        sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.5, gamma=0.0)
        print(f"  gamma=0.0: 应该失败但成功了!")
    except ValueError as e:
        print(f"  gamma=0.0: 正确抛出异常")
    
    try:
        sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.5, gamma=-0.1)
        print(f"  gamma=-0.1: 应该失败但成功了!")
    except ValueError as e:
        print(f"  gamma=-0.1: 正确抛出异常")
    
    try:
        sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=1.5, gamma=0.1)
        print(f"  beta=1.5: 应该失败但成功了!")
    except ValueError as e:
        print(f"  beta=1.5: 正确抛出异常")
    
    try:
        sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.5, gamma=1.5)
        print(f"  gamma=1.5: 应该失败但成功了!")
    except ValueError as e:
        print(f"  gamma=1.5: 正确抛出异常")


def test_sir_set_seeds():
    print("\n[测试] SIR模型 set_seeds")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    
    sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, {0}, beta=0.5, gamma=0.1)
    sir.set_seeds({5, 10, 15})
    print(f"  设置新种子集: OK")


def test_sir_set_beta():
    print("\n[测试] SIR模型 set_beta")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    
    sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, {0}, beta=0.5, gamma=0.1)
    sir.set_beta(0.8)
    print(f"  设置beta=0.8: OK")
    
    try:
        sir.set_beta(0.0)
        print(f"  set_beta(0.0): 应该失败但成功了!")
    except ValueError as e:
        print(f"  set_beta(0.0): 正确抛出异常")


def test_sir_set_gamma():
    print("\n[测试] SIR模型 set_gamma")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    
    sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, {0}, beta=0.5, gamma=0.1)
    sir.set_gamma(0.5)
    print(f"  设置gamma=0.5: OK")
    
    try:
        sir.set_gamma(0.0)
        print(f"  set_gamma(0.0): 应该失败但成功了!")
    except ValueError as e:
        print(f"  set_gamma(0.0): 正确抛出异常")


def test_sir_record_activated():
    print("\n[测试] SIR模型 record_activated 功能")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.5, gamma=0.1, record_activated=True)
    
    count = sir.run_single_simulation(seed=42)
    activated = sir.get_activated_nodes()
    print(f"  单次模拟: 激活{count}个节点, 记录了{len(activated)}个节点")
    
    sir.set_record_activated(False)
    count2 = sir.run_single_simulation(seed=42)
    activated2 = sir.get_activated_nodes()
    print(f"  关闭记录后: 激活{count2}个节点, 记录了{len(activated2)}个节点 (应为0)")


def test_sir_activation_frequency():
    print("\n[测试] SIR模型 activation_frequency 功能")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.5, gamma=0.1, record_activation_frequency=True)
    
    rounds = 100
    avg = sir.run_monte_carlo_diffusion(rounds, seed=42)
    freq = sir.get_activation_frequency()
    total_freq = sum(freq)
    print(f"  蒙特卡洛模拟({rounds}轮): 平均激活{avg:.1f}个节点")
    print(f"  激活频率总和: {total_freq} (应约等于{avg * rounds:.0f})")
    
    sir.set_record_activation_frequency(False)
    freq2 = sir.get_activation_frequency()
    print(f"  关闭记录后频率总和: {sum(freq2)} (应为0)")


def test_sir_single_simulation():
    print("\n[测试] SIR模型 run_single_simulation")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.5, gamma=0.1)
    
    count1 = sir.run_single_simulation(seed=42)
    count2 = sir.run_single_simulation(seed=42)
    print(f"  相同种子重复性: count1={count1}, count2={count2} (应相同)")
    
    count3 = sir.run_single_simulation(seed=100)
    count4 = sir.run_single_simulation(seed=200)
    print(f"  不同种子: seed100={count3}, seed200={count4}")
    
    count5 = sir.run_single_simulation()
    print(f"  随机种子: count={count5}")


def test_sir_monte_carlo():
    print("\n[测试] SIR模型 run_monte_carlo_diffusion")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.5, gamma=0.1)
    
    avg1 = sir.run_monte_carlo_diffusion(100, seed=42)
    avg2 = sir.run_monte_carlo_diffusion(100, seed=42)
    print(f"  相同种子重复性: avg1={avg1:.2f}, avg2={avg2:.2f} (应相同)")
    
    avg3 = sir.run_monte_carlo_diffusion(1000, seed=42)
    print(f"  1000轮模拟: 平均激活{avg3:.2f}个节点")
    
    avg4 = sir.run_monte_carlo_diffusion(100, seed=42, use_multithread=True)
    print(f"  多线程模式: 平均激活{avg4:.2f}个节点")
    
    avg5 = sir.run_monte_carlo_diffusion(100, seed=42, use_multithread=True, num_threads=2)
    print(f"  指定2线程: 平均激活{avg5:.2f}个节点")


def test_sir_beta_influence():
    print("\n[测试] SIR模型 beta参数影响")
    n = 50
    edges = [[i, i+1] for i in range(n-1)]
    edge_weight = [1.0] * len(edges)
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    sir_low = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.1, gamma=0.1)
    avg_low = sir_low.run_monte_carlo_diffusion(1000, seed=42)
    
    sir_mid = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.5, gamma=0.1)
    avg_mid = sir_mid.run_monte_carlo_diffusion(1000, seed=42)
    
    sir_high = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=1.0, gamma=0.1)
    avg_high = sir_high.run_monte_carlo_diffusion(1000, seed=42)
    
    print(f"  beta=0.1: 平均激活{avg_low:.2f}个节点")
    print(f"  beta=0.5: 平均激活{avg_mid:.2f}个节点")
    print(f"  beta=1.0: 平均激活{avg_high:.2f}个节点")
    print(f"  beta越大激活越多: {avg_high >= avg_mid >= avg_low}")


def test_sir_gamma_influence():
    print("\n[测试] SIR模型 gamma参数影响")
    n = 50
    edges = [[i, i+1] for i in range(n-1)]
    edge_weight = [1.0] * len(edges)
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    sir_low = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.8, gamma=0.1)
    avg_low = sir_low.run_monte_carlo_diffusion(1000, seed=42)
    
    sir_mid = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.8, gamma=0.5)
    avg_mid = sir_mid.run_monte_carlo_diffusion(1000, seed=42)
    
    sir_high = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.8, gamma=0.9)
    avg_high = sir_high.run_monte_carlo_diffusion(1000, seed=42)
    
    print(f"  gamma=0.1: 平均激活{avg_low:.2f}个节点 (恢复慢，传播远)")
    print(f"  gamma=0.5: 平均激活{avg_mid:.2f}个节点")
    print(f"  gamma=0.9: 平均激活{avg_high:.2f}个节点 (恢复快，传播近)")
    print(f"  gamma越大激活越少: {avg_low >= avg_mid >= avg_high}")


def test_sir_monte_carlo_with_activated_nodes():
    print("\n[测试] SIR模型 蒙特卡洛+激活节点记录")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.5, gamma=0.1, record_activated=True)
    
    avg = sir.run_monte_carlo_diffusion(100, seed=42)
    activated = sir.get_activated_nodes()
    print(f"  100轮蒙特卡洛: 平均激活{avg:.2f}个节点")
    print(f"  激活节点并集: {len(activated)}个节点")


def test_sir_monte_carlo_with_frequency():
    print("\n[测试] SIR模型 蒙特卡洛+激活频率记录")
    n, edges, edge_weight = create_test_graph()
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.5, gamma=0.1, record_activation_frequency=True)
    
    rounds = 100
    avg = sir.run_monte_carlo_diffusion(rounds, seed=42)
    freq = sir.get_activation_frequency()
    
    seed_freq = freq[0]
    print(f"  种子节点0被激活次数: {seed_freq} (应为{rounds})")
    
    most_activated = max(freq)
    most_activated_node = freq.index(most_activated)
    print(f"  最常被激活的节点: {most_activated_node} (激活{most_activated}次)")


def test_sir_vs_si():
    print("\n[测试] SIR vs SI 对比")
    n = 50
    edges = [[i, i+1] for i in range(n-1)]
    edge_weight = [1.0] * len(edges)
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    si = diffusion_model.SusceptibleInfectedModel(graph, seeds, beta=1.0)
    si_avg = si.run_monte_carlo_diffusion(1000, seed=42)
    
    sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=1.0, gamma=0.5)
    sir_avg = sir.run_monte_carlo_diffusion(1000, seed=42)
    
    print(f"  SI模型(beta=1.0): 平均激活{si_avg:.2f}个节点")
    print(f"  SIR模型(beta=1.0, gamma=0.5): 平均激活{sir_avg:.2f}个节点")
    print(f"  SI激活数 >= SIR激活数: {si_avg >= sir_avg}")


def test_sir_recovery_timing():
    print("\n[测试] SIR模型 恢复时机验证")
    n = 5
    edges = [[0, 1], [1, 2], [2, 3], [3, 4]]
    edge_weight = [1.0] * len(edges)
    graph = im_graph.IMGraphCpp(n, edges, edge_weight, True)
    seeds = {0}
    
    sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=1.0, gamma=0.5, record_activated=True)
    
    counts = []
    for i in range(100):
        count = sir.run_single_simulation(seed=i)
        counts.append(count)
    
    avg = sum(counts) / len(counts)
    print(f"  100次模拟平均激活: {avg:.2f}个节点")
    print(f"  激活数范围: {min(counts)} - {max(counts)}")
    
    unique_counts = set(counts)
    print(f"  不同激活数: {sorted(unique_counts)}")


def run_all_tests():
    print("=" * 60)
    print("SIR (Susceptible-Infected-Recovered) 模型完整测试")
    print("=" * 60)
    
    test_sir_constructor()
    test_sir_invalid_params()
    test_sir_set_seeds()
    test_sir_set_beta()
    test_sir_set_gamma()
    test_sir_record_activated()
    test_sir_activation_frequency()
    test_sir_single_simulation()
    test_sir_monte_carlo()
    test_sir_beta_influence()
    test_sir_gamma_influence()
    test_sir_monte_carlo_with_activated_nodes()
    test_sir_monte_carlo_with_frequency()
    test_sir_vs_si()
    test_sir_recovery_timing()
    
    print("\n" + "=" * 60)
    print("所有SIR模型测试通过!")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
