# PyNetIM

[![PyPI version](https://badge.fury.io/py/pynetim.svg)](https://pypi.org/project/pynetim/)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[PyNetIM](https://zzzkhj.github.io/PyNetIM/) 是一个用于**社交网络影响力最大化（Influence Maximization, IM）**问题的 Python 库，集成了多种经典算法与扩散模型，并提供 **Python 实现 + C++ 加速后端**，适用于算法复现、性能对比与科研实验。

---

## 📦 最新版本
**当前版本**: [v0.4.5](https://github.com/zzzkhj/PyNetIM/releases/tag/v0.4.5)  
**发布日期**: 2026-04-04

### 🎯 v0.4.5 主要更新

#### ✨ 新功能
- **SI 和 SIR 扩散模型 (C++ 后端)**:
  - 新增 `SusceptibleInfectedModel` - SI 传播模型
  - 新增 `SusceptibleInfectedRecoveredModel` - SIR 传播模型
  - 支持参数设置：`beta`（感染概率）、`gamma`（恢复概率）、`max_steps`（最大步数）
  - 支持单次模拟和 Monte Carlo 多轮模拟
  - 支持多线程并行模拟

- **Graph 构造函数支持统一权重**:
  - `IMGraphCpp(num_nodes, edges, weights=1.0)` - 默认所有边权重为 1.0
  - `IMGraphCpp(num_nodes, edges, 0.3)` - 所有边使用统一权重 0.3
  - `IMGraphCpp(num_nodes, edges, [0.1, 0.2, ...])` - 每条边使用各自的权重

#### 📚 API 更新
- **SusceptibleInfectedModel**:
  - 构造函数：`SusceptibleInfectedModel(graph, seeds, beta=0.1, max_steps=100, record_activated=False, record_activation_frequency=False)`
  - `set_beta(beta: float)` - 设置感染概率
  - `set_max_steps(max_steps: int)` - 设置最大传播步数
  - `run_single_simulation(seed=None) -> int` - 单次模拟
  - `run_monte_carlo_diffusion(rounds, seed=None, use_multithread=False, num_threads=0) -> float` - Monte Carlo 模拟

- **SusceptibleInfectedRecoveredModel**:
  - 构造函数：`SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.1, gamma=0.1, record_activated=False, record_activation_frequency=False)`
  - `set_beta(beta: float)` - 设置感染概率
  - `set_gamma(gamma: float)` - 设置恢复概率
  - `run_single_simulation(seed=None) -> int` - 单次模拟
  - `run_monte_carlo_diffusion(rounds, seed=None, use_multithread=False, num_threads=0) -> float` - Monte Carlo 模拟

- **IMGraphCpp**:
  - `weights` 参数现在支持 `float` 类型，表示所有边使用统一权重

#### 🔧 代码重构
- **扩散模型文件分离**:
  - 将扩散模型从单一文件分离到独立文件
  - `independent_cascade.h` - IC 模型
  - `linear_threshold.h` - LT 模型
  - `susceptible_infected.h` - SI 模型
  - `susceptible_infected_recovered.h` - SIR 模型
  - `common.h` - 共享代码（ObjectPool、RNG 工具函数）

#### 🧪 测试
- 新增完整测试覆盖：
  - `test_ic_model.py` - IC 模型完整测试
  - `test_lt_model.py` - LT 模型完整测试
  - `test_si_model.py` - SI 模型完整测试
  - `test_sir_model.py` - SIR 模型完整测试
- 单线程和多线程结果一致性验证通过
- 参数验证测试通过

#### 📝 使用示例
```python
import pynetim.cpp.graph as im_graph
import pynetim.cpp.diffusion_model as diffusion_model

# 创建图（统一权重）
graph = im_graph.IMGraphCpp(100, [(i, i+1) for i in range(99)], 0.3)
seeds = {0}

# SI 模型
si = diffusion_model.SusceptibleInfectedModel(graph, seeds, beta=0.3)
avg = si.run_monte_carlo_diffusion(1000, seed=42)
print(f"SI 平均感染节点数: {avg:.2f}")

# SIR 模型
sir = diffusion_model.SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.3, gamma=0.1)
avg = sir.run_monte_carlo_diffusion(1000, seed=42)
print(f"SIR 平均感染节点数: {avg:.2f}")

# 多线程模拟
avg = sir.run_monte_carlo_diffusion(10000, seed=42, use_multithread=True)
print(f"多线程 SIR 平均感染节点数: {avg:.2f}")
```

📖 **查看完整更新**: [CHANGELOG.md](CHANGELOG.md)

---

## ✨ 功能概览

* 多种经典影响力最大化算法（Heuristic / Simulation / RIS）
* 多种传播模型（IC / LT / SI / SIR）
* 统一的图结构封装（基于 NetworkX）
* Python 可读实现 + C++ 高性能实现
* 支持蒙特卡洛扩散模拟
* 内置算法计时装饰器

---

## 📁 项目结构总览

```
src/
├─ pynetim/
│  ├─ __init__.py
│  │
│  ├─ cpp/                  # C++ 后端（pybind11 绑定）
│  │  ├─ diffusion_model/
│  │  ├─ graph/
│  │  └─ __init__.py
│  │
│  ├─ py/                   # Python 实现（核心逻辑）
│  │  ├─ algorithms/
│  │  ├─ diffusion_model/
│  │  ├─ graph/
│  │  └─ decorator/
│  │
│  ├─ utils/                # 通用工具函数
│  └─ __init__.py
│
└─ tests/                   # 测试代码
```

---

## 🧠 架构说明

PyNetIM 采用 **"Python 逻辑层 + C++ 计算层"** 的设计：

* **算法逻辑、实验流程** → Python
* **高频计算（扩散 / 图操作）** → C++
* 上层算法 **对后端透明**

```
Algorithm (Python)
   ↓
Diffusion Model Interface
   ↓
Graph Interface
   ↓
Python 实现  /  C++ 扩展
```

---

## ⚙️ C++ 后端（`pynetim/cpp`）

该目录包含 **C++ 实现的高性能模块**，通过 `pybind11` 暴露给 Python。

### 📁 `cpp/diffusion_model`

```
cpp/diffusion_model/
├─ independent_cascade_model.pyi  # IC Python 类型存根
├─ linear_threshold_model.pyi     # LT Python 类型存根
├─ susceptible_infected_model.pyi # SI Python 类型存根
├─ sir_model.pyi                  # SIR Python 类型存根
└─ __init__.py
```

支持的模型：

* Independent Cascade (IC)
* Linear Threshold (LT)
* Susceptible-Infected (SI)
* Susceptible-Infected-Recovered (SIR)

---

### 📁 `cpp/graph`

```
cpp/graph/
├─ Graph.h            # C++ 图结构定义
├─ graph_bind.cpp     # 图结构 pybind11 绑定
├─ graph.pyi          # Python 类型存根
└─ __init__.py
```

功能：

* 提供 C++ 层图结构
* 为扩散模型提供高效邻接访问

---

## 🐍 Python 实现（`pynetim/py`）

该部分包含 **完整、可读、可修改的实现**，是算法理解与二次开发的主要入口。

---

### 📁 `py/algorithms` —— 影响力最大化算法

```
py/algorithms/
├─ base_algorithm.py
├─ heuristic_algorithm.py
├─ simulation_algorithm.py
├─ RIS_algorithm.py
└─ __init__.py
```

#### 已实现算法

**启发式算法**（速度快）：

* `SingleDiscountAlgorithm`
* `DegreeDiscountAlgorithm`

**基于模拟的算法**（精度高）：

* `GreedyAlgorithm`
* `CELFAlgorithm`

**RIS 系列算法**（适合大规模图）：

* `BaseRISAlgorithm`
* `IMMAlgorithm`

---

### 📁 `py/diffusion_model` —— 扩散模型（Python 版本）

```
py/diffusion_model/
├─ base_diffusion_model.py
├─ independent_cascade_model.py
├─ linear_threshold_model.py
├─ susceptible_infected_model.py
├─ susceptible_infected_recovered_model.py
├─ run_monte_carlo_diffusion.py
└─ __init__.py
```

---

### 📁 `py/graph` —— Python 图封装

```
py/graph/
├─ graph.py
└─ __init__.py
```

核心类：

* **IMGraph**
  * 封装 NetworkX 图
  * 管理节点、边、权重
  * 为算法与扩散模型提供统一接口

---

### 📁 `py/decorator` —— 装饰器

```
py/decorator/
├─ decorator.py
└─ __init__.py
```

* `Timer`：用于统计算法运行时间

---

## 🧰 工具函数（`pynetim/utils`）

```
utils/
├─ utils.py
└─ __init__.py
```

主要功能：

* `set_edge_weight`
  * WC（入度倒数）
  * 随机权重
  * 自定义权重


---

## 🧪 测试

### 运行测试

PyNetIM 提供了完整的测试套件，用于验证库的功能和性能。

#### NetworkX 图测试
运行 NetworkX 图的测试：

```bash
conda run -n pynetim python tests/test.py
```

#### C++ 图测试
运行 C++ 图的测试：

```bash
conda run -n pynetim python tests/test_cpp.py
```

### 测试覆盖

测试套件涵盖了以下功能：

**NetworkX 图测试** (`tests/test.py`):
- 基本图操作（创建、节点、边）
- 图统计信息计算
- 图密度计算
- 连通性分析
- 边权重设置（CONSTANT、TV、WC）
- 感染阈值计算

**C++ 图测试** (`tests/test_cpp.py`):
- 基本图操作（创建、节点、边）
- 邻接查询
- 度数查询（批量）
- 边权重设置
- IC 扩散模型测试
- LT 扩散模型测试
- SI 扩散模型测试
- SIR 扩散模型测试

---

## 🚀 快速开始

```python
import networkx as nx
from pynetim.py.graph import IMGraph
from pynetim.py.algorithms import DegreeDiscountAlgorithm

# 创建图
g = nx.erdos_renyi_graph(100, 0.1)

# 构建 IMGraph（WC 权重）
graph = IMGraph(g, edge_weight_type='WC')

# 运行算法
algo = DegreeDiscountAlgorithm(graph)
seeds = algo.run(k=10)

print(seeds)
```

---

## 🔧 扩展说明

* ✔ 可新增 **Python 扩散模型**
* ✔ 可新增 **自定义 IM 算法**
* ✔ C++ 层主要用于性能优化

---

## 📦 安装

```bash
pip install pynetim
```

---

## 🙏 致谢

感谢以下对本项目提供帮助的个人和工具：

- **TraeAI** - 提供了强大的 AI 辅助开发环境，显著提升了代码开发效率和问题解决能力
