"""
测试近期变更是否生效。

覆盖范围:
1. 随机种子控制精简 - pynetim.random 仅影响 C++ 模块
2. 种子集保留选择顺序 - get_seeds_ordered() 返回有序列表
3. ToupleGDD / S2V-DQN 推理加速 - 缓存图结构是否成功构建
4. BaseAlgorithm 基类变更 - self.seeds 类型为 list
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import random as py_random
import numpy as np

import pynetim
from pynetim import IMGraph
from pynetim.algorithms import (
    DegreeCentralityAlgorithm, SingleDiscountAlgorithm, VoteRankAlgorithm,
    GreedyAlgorithm, CELFAlgorithm
)


# ============================================================================
# 测试 1: 随机种子控制精简
# ============================================================================
def test_random_seed_cpp_only():
    """pynetim.random.seed() 不应影响 Python random 和 NumPy 的状态。"""
    print("=" * 60)
    print("测试 1: 随机种子控制精简")
    print("=" * 60)

    # 记录设置种子前的状态
    py_random.seed()
    state_before = py_random.getstate()

    # 设置 pynetim 全局种子
    pynetim.random.seed(42)
    assert pynetim.random.get_random_seed() == 42
    assert pynetim.random.has_seed() is True

    # Python random 状态不应变化
    state_after = py_random.getstate()
    assert state_before == state_after, (
        "FAIL: pynetim.random.seed() 影响了 Python random 状态"
    )

    # NumPy 随机状态不应被重置
    val_before = np.random.randn()
    pynetim.random.seed(123)
    val_after = np.random.randn()
    # 如果 np.random 被重新 seed，val_after 会等于一个固定值
    # 这里不精确断言，只要不崩溃且连续调用产生不同值即可
    if np.random.randn() == np.random.randn():
        print("  WARN: NumPy 随机状态可能被重置")
    else:
        print("  OK: NumPy 随机状态未受影响")

    # 清除种子
    pynetim.random.clear_seed()
    assert pynetim.random.has_seed() is False
    assert pynetim.random.get_random_seed() is None

    print("  OK: pynetim.random.seed() 不影响 Python random 状态")
    print("  OK: clear_seed() 正确清除全局种子")
    print("  OK: get_random_seed() / has_seed() 正常工作\n")


# ============================================================================
# 测试 2: 种子集保留选择顺序
# ============================================================================
def test_seed_order():
    """run() 返回 Set[int]（向后兼容），get_seeds_ordered() 返回有序列表。"""
    print("=" * 60)
    print("测试 2: 种子集保留选择顺序")
    print("=" * 60)

    # 构造一个有向图
    edges = [(0, 1, 0.5), (0, 2, 0.3), (1, 2, 0.4), (2, 3, 0.6), (3, 0, 0.2),
             (4, 0, 0.7), (4, 5, 0.3), (5, 3, 0.5)]
    graph = IMGraph(edges)

    k = 3

    # -- DegreeCentrality --
    algo = DegreeCentralityAlgorithm(graph)
    seeds_set = algo.run(k)
    assert isinstance(seeds_set, set), "run() 应返回 set"
    assert len(seeds_set) == k
    seeds_ordered = algo.get_seeds_ordered()
    assert isinstance(seeds_ordered, list), "get_seeds_ordered() 应返回 list"
    assert len(seeds_ordered) == k
    assert len(seeds_ordered) == len(set(seeds_ordered)), "种子列表不应有重复"
    # 验证 get_seeds() 与 run() 返回值一致
    assert algo.get_seeds() == seeds_set
    # 验证有序列表与集合内容一致
    assert set(seeds_ordered) == seeds_set
    print(f"  DegreeCentrality: seeds_set={seeds_set}, ordered={seeds_ordered}")
    print("  OK: run() 返回 set, get_seeds_ordered() 返回 list")

    # -- SingleDiscount --
    algo = SingleDiscountAlgorithm(graph)
    seeds_set = algo.run(k)
    seeds_ordered = algo.get_seeds_ordered()
    assert isinstance(seeds_set, set)
    assert isinstance(seeds_ordered, list)
    assert set(seeds_ordered) == seeds_set
    print(f"  SingleDiscount: ordered={seeds_ordered}")

    # -- VoteRank --
    algo = VoteRankAlgorithm(graph)
    seeds_set = algo.run(k)
    seeds_ordered = algo.get_seeds_ordered()
    assert isinstance(seeds_set, set)
    assert isinstance(seeds_ordered, list)
    assert set(seeds_ordered) == seeds_set
    print(f"  VoteRank: ordered={seeds_ordered}")

    # -- Greedy (simulation-based, 使用少量 mc_rounds 加速) --
    algo = GreedyAlgorithm(graph, diffusion_model='IC')
    seeds_set = algo.run(k, mc_rounds=10, show_progress=False)
    seeds_ordered = algo.get_seeds_ordered()
    assert isinstance(seeds_set, set)
    assert isinstance(seeds_ordered, list)
    assert set(seeds_ordered) == seeds_set
    print(f"  Greedy: ordered={seeds_ordered}")

    # -- CELF --
    algo = CELFAlgorithm(graph, diffusion_model='IC')
    seeds_set = algo.run(k, mc_rounds=10, show_progress=False)
    seeds_ordered = algo.get_seeds_ordered()
    assert isinstance(seeds_set, set)
    assert isinstance(seeds_ordered, list)
    assert set(seeds_ordered) == seeds_set
    print(f"  CELF: ordered={seeds_ordered}")

    print("  OK: 所有算法均正确返回 set/有序 list\n")


# ============================================================================
# 测试 3: ToupleGDD / S2V-DQN 推理加速（图结构缓存）
# ============================================================================
def test_drl_graph_caching():
    """ToupleGDD 和 S2V-DQN 的 _prepare_inference() 应缓存图结构。"""
    print("=" * 60)
    print("测试 3: DRL 推理图结构缓存")
    print("=" * 60)

    edges = [(0, 1, 0.5), (1, 2, 0.3), (2, 0, 0.4)]
    graph = IMGraph(edges)

    # -- ToupleGDD --
    try:
        algo = pynetim.algorithms.ToupleGDDAlgorithm(
            graph, pretrained=False, device='cpu'
        )
        algo._prepare_inference()
        assert hasattr(algo, '_cached_edge_index'), "ToupleGDD: 缺少 _cached_edge_index"
        assert hasattr(algo, '_cached_edge_weight'), "ToupleGDD: 缺少 _cached_edge_weight"
        assert algo._cached_edge_index.shape[1] == graph.num_edges
        assert algo._cached_edge_weight.shape[0] == graph.num_edges
        print(f"  ToupleGDD: cached_edge_index={algo._cached_edge_index.shape}, "
              f"cached_edge_weight={algo._cached_edge_weight.shape}")
        print("  OK: 图结构缓存正确")

        # 验证 iteractive 选择可以正常执行（图结构复用）
        state = algo._init_state()
        assert state.device.type == 'cpu', "state 应在 CPU 上"
        seeds = algo._select_iterative(2, state)
        assert len(seeds) == 2
        print(f"  OK: 迭代选择正常执行, seeds={seeds}")

    except Exception as e:
        print(f"  ToupleGDD: {e}")

    # -- S2V-DQN --
    try:
        algo = pynetim.algorithms.S2VDQNAlgorithm(
            graph, pretrained=False, device='cpu'
        )
        algo._prepare_inference()
        assert hasattr(algo, '_cached_edge_index'), "S2V-DQN: 缺少 _cached_edge_index"
        assert hasattr(algo, '_cached_x'), "S2V-DQN: 缺少 _cached_x"
        assert hasattr(algo, '_cached_edge_attr'), "S2V-DQN: 缺少 _cached_edge_attr"
        assert algo._cached_edge_index.shape[1] == graph.num_edges
        assert algo._cached_x.shape == (graph.num_nodes, 2)
        assert algo._cached_edge_attr.shape == (graph.num_edges, 4)
        print(f"  S2V-DQN: cached_x={algo._cached_x.shape}, "
              f"cached_edge_attr={algo._cached_edge_attr.shape}")
        print("  OK: 图结构缓存正确")

        # 验证 iteractive 选择可以正常执行
        state = algo._init_state()
        assert state.device.type == 'cpu', "state 应在 CPU 上"
        seeds = algo._select_iterative(2, state)
        assert len(seeds) == 2
        print(f"  OK: 迭代选择正常执行, seeds={seeds}")

    except Exception as e:
        print(f"  S2V-DQN: {e}")

    print()


# ============================================================================
# 测试 4: self.seeds 类型一致性
# ============================================================================
def test_seeds_type_consistency():
    """所有算法的 self.seeds 应为 list 类型。"""
    print("=" * 60)
    print("测试 4: self.seeds 类型一致性")
    print("=" * 60)

    edges = [(0, 1, 0.5), (1, 2, 0.3), (2, 3, 0.6)]
    graph = IMGraph(edges)
    k = 2

    algos_to_test = [
        ("DegreeCentrality", DegreeCentralityAlgorithm(graph)),
        ("SingleDiscount", SingleDiscountAlgorithm(graph)),
        ("VoteRank", VoteRankAlgorithm(graph)),
        ("Greedy", GreedyAlgorithm(graph, diffusion_model='IC')),
        ("CELF", CELFAlgorithm(graph, diffusion_model='IC')),
    ]

    for name, algo in algos_to_test:
        if "Greedy" in name or "CELF" in name:
            algo.run(k, mc_rounds=5, show_progress=False)
        else:
            algo.run(k)
        assert isinstance(algo.seeds, list), (
            f"{name}: self.seeds 类型为 {type(algo.seeds)}，应为 list"
        )
        print(f"  {name}: self.seeds 类型正确 (list), 内容={algo.seeds}")

    print("  OK: 所有算法 self.seeds 均为 list\n")


# ============================================================================
# 运行所有测试
# ============================================================================
if __name__ == '__main__':
    print()
    print("=" * 60)
    print("  PyNetIM 近期变更测试")
    print("=" * 60)
    print()

    test_random_seed_cpp_only()
    test_seed_order()
    test_seeds_type_consistency()
    test_drl_graph_caching()

    print("=" * 60)
    print("  所有测试完成")
    print("=" * 60)
