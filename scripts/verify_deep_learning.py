#!/usr/bin/env python
"""深度学习模块功能验证脚本。

验证所有导出的算法类和训练器类都可以正常使用。
"""

import sys
sys.path.insert(0, 'src')


def test_imports():
    """测试所有组件可以正常导入。"""
    print('=' * 60)
    print('1. 测试导入')
    print('=' * 60)

    from pynetim.algorithms.deep_learning import (
        ToupleGDDAlgorithm, S2VDQNAlgorithm,
        ToupleGDDTrainer, S2VDQNTrainer,
        BiGDNAlgorithm, BiGDNSAlgorithm, BiGDNTrainer,
        BiGDNNodeEncoderTrainer
    )

    print('✓ ToupleGDDAlgorithm')
    print('✓ S2VDQNAlgorithm')
    print('✓ BiGDNAlgorithm')
    print('✓ BiGDNSAlgorithm')
    print('✓ ToupleGDDTrainer')
    print('✓ S2VDQNTrainer')
    print('✓ BiGDNTrainer')
    print('✓ BiGDNNodeEncoderTrainer')

    return True


def test_algorithm_creation():
    """测试算法类可以正常创建实例。"""
    print()
    print('=' * 60)
    print('2. 测试算法类创建')
    print('=' * 60)

    from pynetim import IMGraph
    from pynetim.algorithms.deep_learning import (
        ToupleGDDAlgorithm, S2VDQNAlgorithm,
        BiGDNAlgorithm, BiGDNSAlgorithm
    )

    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
             (0, 2), (1, 3), (2, 4), (3, 5)]
    graph = IMGraph(edges, weights=0.3)

    algo1 = ToupleGDDAlgorithm(graph, pretrained=True, device='cpu')
    print(f'✓ ToupleGDDAlgorithm 创建成功: {type(algo1).__name__}')

    algo2 = S2VDQNAlgorithm(graph, pretrained=True, device='cpu')
    print(f'✓ S2VDQNAlgorithm 创建成功: {type(algo2).__name__}')

    algo3 = BiGDNAlgorithm(graph, pretrained=True, device='cpu')
    print(f'✓ BiGDNAlgorithm 创建成功: {type(algo3).__name__}')

    algo4 = BiGDNSAlgorithm(graph, teacher_path=None, pretrained=True, device='cpu')
    print(f'✓ BiGDNSAlgorithm 创建成功: {type(algo4).__name__}')

    return True


def test_algorithm_inference():
    """测试算法推理功能。"""
    print()
    print('=' * 60)
    print('3. 测试算法推理')
    print('=' * 60)

    from pynetim import IMGraph
    from pynetim.algorithms.deep_learning import (
        ToupleGDDAlgorithm, S2VDQNAlgorithm,
        BiGDNAlgorithm
    )

    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
             (0, 2), (1, 3), (2, 4), (3, 5)]
    graph = IMGraph(edges, weights=0.3)

    algo1 = ToupleGDDAlgorithm(graph, pretrained=True, device='cpu')
    seeds1 = algo1.run(k=2)
    print(f'✓ ToupleGDDAlgorithm 推理成功: seeds={list(seeds1)}')

    algo2 = S2VDQNAlgorithm(graph, pretrained=True, device='cpu')
    seeds2 = algo2.run(k=2)
    print(f'✓ S2VDQNAlgorithm 推理成功: seeds={list(seeds2)}')

    algo3 = BiGDNAlgorithm(graph, pretrained=True, device='cpu')
    seeds3 = algo3.run(k=2)
    print(f'✓ BiGDNAlgorithm 推理成功: seeds={list(seeds3)}')

    return True


def test_trainer_creation():
    """测试训练器类可以正常创建实例。"""
    print()
    print('=' * 60)
    print('4. 测试训练器类创建')
    print('=' * 60)

    from pynetim.algorithms.deep_learning import (
        ToupleGDDTrainer, S2VDQNTrainer,
        BiGDNTrainer, BiGDNNodeEncoderTrainer
    )

    trainer1 = ToupleGDDTrainer(device='cpu')
    print(f'✓ ToupleGDDTrainer 创建成功')
    print(f'  - model_type: {trainer1.model_type}')
    print(f'  - model class: {type(trainer1.model).__name__}')

    trainer2 = S2VDQNTrainer(device='cpu')
    print(f'✓ S2VDQNTrainer 创建成功')
    print(f'  - model_type: {trainer2.model_type}')
    print(f'  - model class: {type(trainer2.model).__name__}')

    trainer3 = BiGDNTrainer(num_features=64, device='cpu')
    print(f'✓ BiGDNTrainer 创建成功')

    trainer4 = BiGDNNodeEncoderTrainer(num_features=64, device='cpu')
    print(f'✓ BiGDNNodeEncoderTrainer 创建成功')

    return True


def test_class_independence():
    """测试类独立性。"""
    print()
    print('=' * 60)
    print('5. 测试类独立性')
    print('=' * 60)

    from pynetim.algorithms.deep_learning import (
        ToupleGDDTrainer, S2VDQNTrainer,
        BiGDNTrainer, BiGDNNodeEncoderTrainer
    )

    result1 = ToupleGDDTrainer is S2VDQNTrainer
    print(f'ToupleGDDTrainer is S2VDQNTrainer: {result1}')
    assert result1 == False, 'ToupleGDDTrainer 和 S2VDQNTrainer 应该是独立的类'

    result2 = BiGDNTrainer is BiGDNNodeEncoderTrainer
    print(f'BiGDNTrainer is BiGDNNodeEncoderTrainer: {result2}')
    assert result2 == False, 'BiGDNTrainer 和 BiGDNNodeEncoderTrainer 应该是独立的类'

    print('✓ 所有类都是独立的')

    return True


def test_deleted_components():
    """验证已删除的组件确实不存在。"""
    print()
    print('=' * 60)
    print('6. 验证已删除的组件')
    print('=' * 60)

    try:
        from pynetim.algorithms.deep_learning import pretrain_node_encoder
        print('❌ pretrain_node_encoder 仍然存在！')
        return False
    except ImportError:
        print('✓ pretrain_node_encoder 已删除')

    try:
        from pynetim.algorithms.deep_learning import Trainer
        print('❌ Trainer 仍然存在！')
        return False
    except ImportError:
        print('✓ Trainer 已删除')

    return True


def main():
    """运行所有测试。"""
    print()
    print('*' * 60)
    print('*  深度学习模块功能验证')
    print('*' * 60)
    print()

    tests = [
        ('导入测试', test_imports),
        ('算法类创建测试', test_algorithm_creation),
        ('算法推理测试', test_algorithm_inference),
        ('训练器类创建测试', test_trainer_creation),
        ('类独立性测试', test_class_independence),
        ('已删除组件验证', test_deleted_components),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f'❌ {name} 失败')
        except Exception as e:
            failed += 1
            print(f'❌ {name} 异常: {e}')

    print()
    print('=' * 60)
    print('测试结果')
    print('=' * 60)
    print(f'通过: {passed}/{len(tests)}')
    print(f'失败: {failed}/{len(tests)}')

    if failed == 0:
        print()
        print('✓ 所有测试通过!')
        return 0
    else:
        print()
        print('❌ 部分测试失败!')
        return 1


if __name__ == '__main__':
    sys.exit(main())
