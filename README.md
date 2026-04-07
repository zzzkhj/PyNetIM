# PyNetIM

[![PyPI version](https://badge.fury.io/py/pynetim.svg)](https://pypi.org/project/pynetim/)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[PyNetIM](https://zzzkhj.github.io/PyNetIM/) 是一个用于**社交网络影响力最大化（Influence Maximization, IM）问题**的 Python 库，集成了多种经典算法与扩散模型，提供 **C++ 高性能后端**，适用于算法研究、性能对比与科研实验。

***

## 简介

PyNetIM 提供完整的影响力最大化解决方案：

- **多种传播模型** - IC、LT、SI、SIR
- **多种 IM 算法** - 启发式、模拟类、RIS 类、OPIM 类
- **高性能 C++ 后端** - 比纯 Python 快 20-30 倍
- **自定义模型支持** - 支持用户自定义传播模型
- **简洁 API** - 一行代码完成复杂任务

***

## 安装

```bash
pip install pynetim
```

**系统要求：**
- Python 3.8+（推荐 3.10+）
- C++20 编译器（GCC 10+, Clang 10+, MSVC 19.28+）

***

## 快速开始

```python
from pynetim import IMGraph, IndependentCascadeModel, IMMAlgorithm

# 1. 创建图
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
graph = IMGraph(edges, weights=0.3)

# 2. 使用 IMM 算法选择种子节点
imm = IMMAlgorithm(graph, model='IC', epsilon=0.5)
seeds = imm.run(k=2)
print(f"种子节点: {seeds}")

# 3. 评估影响力
model = IndependentCascadeModel(graph, set(seeds))
avg_influence = model.run_monte_carlo_diffusion(1000, num_threads=4)
print(f"平均影响力: {avg_influence:.2f}")
```

***

## 核心功能

### 图结构

```python
from pynetim import IMGraph

# 从边列表创建
graph = IMGraph(edges, weights=0.3)

# 统一权重
graph = IMGraph(edges, weights=0.1)

# 逐边权重
graph = IMGraph(edges, weights=[0.1, 0.2, 0.3, ...])

# 查询邻居（仅节点ID）
for neighbor in graph.out_neighbors(node):
    print(f"邻居: {neighbor}")

# 查询邻居（带权重）
for neighbor, weight in graph.out_neighbors_with_weights(node):
    print(f"邻居: {neighbor}, 权重: {weight}")
```

### 传播模型

| 模型                                  | 说明     | 使用场景      |
| ----------------------------------- | ------ | --------- |
| `IndependentCascadeModel`           | 独立级联模型 | 社交网络传播    |
| `LinearThresholdModel`              | 线性阈值模型 | 观点传播      |
| `SusceptibleInfectedModel`          | SI 模型  | 疫情传播      |
| `SusceptibleInfectedRecoveredModel` | SIR 模型 | 疫情传播（含恢复） |

```python
from pynetim import IndependentCascadeModel, SusceptibleInfectedModel, SusceptibleInfectedRecoveredModel

# IC 模型
model = IndependentCascadeModel(graph, seeds={0, 1})
count = model.run_single_simulation()

# SI 模型（需要显式提供 beta 和 max_steps）
si_model = SusceptibleInfectedModel(graph, seeds={0, 1}, beta=0.1, max_steps=100)
count = si_model.run_single_simulation()

# SIR 模型（需要显式提供 beta 和 gamma）
sir_model = SusceptibleInfectedRecoveredModel(graph, seeds={0, 1}, beta=0.1, gamma=0.05)
count = sir_model.run_single_simulation()

# 蒙特卡洛模拟（多线程）
avg = model.run_monte_carlo_diffusion(1000, num_threads=4)

# 记录激活节点
model = IndependentCascadeModel(graph, seeds, record_activated=True)
count = model.run_single_simulation()
activated = model.get_activated_nodes()
```

### 影响力最大化算法

| 算法                        | 类型    | 特点       | 参考文献 |
| ------------------------- | ----- | -------- | ------ |
| `SingleDiscountAlgorithm` | 启发式   | 速度快      | - |
| `DegreeDiscountAlgorithm` | 启发式   | 速度快，效果好  | - |
| `GreedyAlgorithm`         | 模拟类   | 精度高，速度慢  | Kemper et al., 2003 |
| `CELFAlgorithm`           | 模拟类   | 精度高，比贪婪快 | Leskovec et al., 2007 |
| `CELFPlusAlgorithm`       | 模拟类   | CELF 优化版 | Goyal et al., WWW 2011 |
| `BaseRISAlgorithm`        | RIS 类 | 基础反向影响采样 | Borgs et al., 2014 |
| `IMMAlgorithm`            | RIS 类 | 大规模图首选   | Tang et al., SIGMOD 2015 |
| `TIMAlgorithm`            | RIS 类 | 两阶段影响力估计 | Tang et al., 2014 |
| `TIMPlusAlgorithm`        | RIS 类 | TIM 优化版  | Tang et al., 2014 |
| `OPIMAlgorithm`           | OPIM 类 | 可证明近似保证  | Tang et al., SIGMOD 2018 |
| `OPIMCAlgorithm`          | OPIM 类 | 自适应采样版本 | Tang et al., SIGMOD 2018 |

```python
from pynetim import DegreeDiscountAlgorithm, GreedyAlgorithm, IMMAlgorithm, OPIMCAlgorithm

# 启发式算法（快）
algo = DegreeDiscountAlgorithm(graph)
seeds = algo.run(k=10)

# 模拟类算法（精确）
algo = GreedyAlgorithm(graph, model='IC', num_trials=1000)
seeds = algo.run(k=10)

# RIS 算法（大规模图）
algo = IMMAlgorithm(graph, model='IC', epsilon=0.5)
seeds = algo.run(k=10)

# OPIM-C 算法（自适应采样，可证明近似保证）
algo = OPIMCAlgorithm(graph, model='IC', random_seed=42, verbose=True)
seeds = algo.run(k=10, epsilon=0.3)
```

### 自定义传播模型

PyNetIM 提供两种自定义模型基类：

| 基类                               | 特点         | 适用场景      |
| -------------------------------- | ---------- | --------- |
| `BaseCallbackDiffusionModel`     | C++ 回调，单线程 | 简单模型      |
| `BaseMultiprocessDiffusionModel` | 多进程并行      | 需要加速的复杂模型 |

```python
from pynetim.diffusion_model import BaseMultiprocessDiffusionModel

class MyICModel(BaseMultiprocessDiffusionModel):
    """自定义 IC 模型。"""
    
    def run_single_trial(self, seeds, rng_seed):
        import random
        random.seed(rng_seed)
        
        activated = set(seeds)
        current = list(seeds)
        
        while current:
            new_active = []
            for node in current:
                for neighbor, weight in self.graph.out_neighbors_with_weights(node):
                    if neighbor not in activated and random.random() < weight:
                        activated.add(neighbor)
                        new_active.append(neighbor)
            current = new_active
        
        return len(activated), activated, [0] * self.graph.num_nodes

# 使用自定义模型
model = MyICModel(graph, {0, 1})
avg = model.run_monte_carlo_diffusion(1000, num_processes=4)
```

***

## 性能对比

测试环境：Intel Xeon Platinum 8255C @ 2.50GHz, 3.6GB RAM, Ubuntu 24.04 LTS, Python 3.10.20

| 模型                             | 并行度  | 耗时    | 说明        |
| ------------------------------ | ---- | ----- | --------- |
| C++ IC                         | 1 线程 | 0.30s | 基准（最快）    |
| C++ IC                         | 4 线程 | 0.38s | 多线程加速     |
| BaseCallbackDiffusionModel     | 1 线程 | 6.6s  | 单线程可用     |
| BaseCallbackDiffusionModel     | 4 线程 | 6.9s  | ⚠️ GIL 限制 |
| BaseMultiprocessDiffusionModel | 1 进程 | 6.2s  | 单进程       |
| BaseMultiprocessDiffusionModel | 4 进程 | 3.7s  | ✅ 真正并行    |

***

## 项目结构

```
src/pynetim/
├── __init__.py
├── graph/                    # 图结构
│   └── IMGraph
├── diffusion_model/          # 传播模型
│   ├── IndependentCascadeModel
│   ├── LinearThresholdModel
│   ├── SusceptibleInfectedModel
│   ├── SusceptibleInfectedRecoveredModel
│   ├── BaseCallbackDiffusionModel      # C++ 回调版
│   └── BaseMultiprocessDiffusionModel  # 多进程版
├── algorithms/               # IM 算法
│   ├── SingleDiscountAlgorithm
│   ├── DegreeDiscountAlgorithm
│   ├── GreedyAlgorithm
│   ├── CELFAlgorithm
│   ├── ris/                  # RIS 类算法
│   │   ├── BaseRISAlgorithm
│   │   ├── IMMAlgorithm
│   │   ├── TIMAlgorithm
│   │   ├── TIMPlusAlgorithm
│   │   ├── OPIMAlgorithm
│   │   └── OPIMCAlgorithm
├── utils/                    # 工具函数
└── py/                       # Python 实现（维护模式）
```

***

## 开发指南

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python tests/test_diffusion_comparison.py
```

### 贡献代码

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

***

## 更新日志

查看 [CHANGELOG.md](CHANGELOG.md) 了解版本历史。

***

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

***

## 致谢

感谢以下对本项目提供帮助的个人和工具：

- **TraeAI** - 提供了强大的 AI 辅助开发环境，显著提升了代码开发效率和问题解决能力
- **GLM-5** - 智谱 AI 大语言模型，在代码开发、调试优化和文档编写过程中提供了重要帮助
