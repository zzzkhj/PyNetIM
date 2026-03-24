# PyNetIM

[![PyPI version](https://badge.fury.io/py/pynetim.svg)](https://pypi.org/project/pynetim/)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[PyNetIM](https://zzzkhj.github.io/PyNetIM/) 是一个用于**社交网络影响力最大化（Influence Maximization, IM）**问题的 Python 库，集成了多种经典算法与扩散模型，并提供 **Python 实现 + C++ 加速后端**，适用于算法复现、性能对比与科研实验。

---

## 📦 最新版本
**当前版本**: [v0.4.2](https://github.com/zzzkhj/PyNetIM/releases/tag/v0.4.2)  
**发布日期**: 2026-03-24

### 🎯 v0.4.2 主要更新

#### 🐛 Bug 修复
- **修复 Graph 对象生命周期管理问题**:
  - 修复了 Python 层 Graph 对象被垃圾回收后，C++ 层模型仍持有悬空指针导致的段错误
  - 根本原因：Graph 没有以正确的 `std::shared_ptr` 形式被 Python 管理
  - 解决方案：在 [graph_bind.cpp](src/pynetim/cpp/bindings/graph_bind.cpp) 中使用 `std::make_shared` 创建 Graph 对象
  - 现在 Python 和 C++ 共享同一个 `shared_ptr`，引用计数统一管理
  - `py::keep_alive` 真正生效，确保 Graph 对象在模型使用期间不会被回收

#### 🧪 测试
- 新增生命周期验证测试：
  - `test_weakref.py` - 使用 weakref 验证 Graph 对象生命周期
  - `test_weakref2.py` - 带 seeds 的生命周期测试
  - 测试结果：Graph 对象在模型使用期间保持存活，不会发生段错误

#### 📚 技术细节
- **关键知识点**: `py::class_<T, std::shared_ptr<T>>` 只保证 Python 用 shared_ptr 管理对象，但前提是这个对象本来就是 shared_ptr 创建的
- **修改文件**: [graph_bind.cpp](src/pynetim/cpp/bindings/graph_bind.cpp#L18-L28)

📖 **查看完整更新**: [CHANGELOG.md](CHANGELOG.md)

---

### 🎯 v0.4.1 主要更新

#### 🐛 Bug 修复
- **修复 C++ 扩展的线程安全问题**:
  - 修复 `IndependentCascadeModel` 和 `LinearThresholdModel` 在多线程模式下的线程安全问题
  - 通过参数传递替代直接访问成员变量 `seeds`，消除跨线程共享 `std::set` 的状态
  - 解决了与 PyTorch 等多线程库结合使用时的 segmentation fault 问题
  - 经过严格测试：5000 次迭代、8 线程并发、10 图并发均稳定运行
  - 单线程和多线程结果完全一致，确保正确性

#### 🧪 测试
- 新增严格验证测试：
  - 单线程/多线程一致性测试
  - 长时间多线程压力测试 (2000 次迭代)
  - 多线程并发访问测试 (8 线程)
  - 多图并发处理测试 (10 个图)
  - 极限压力测试 (5000 次迭代)

📖 **查看完整更新**: [CHANGELOG.md](CHANGELOG.md)

---

### 🎯 v0.4.0 主要更新

#### ✨ 新功能
- **C++ 图完整支持**: 所有工具函数现在支持 `IMGraphCpp`（C++ 图）
  - `set_edge_weight()` - 支持三种权重模型（CONSTANT、TV、WC）
  - `infection_threshold()` - 支持基于度分布的感染阈值计算
  - `graph_statistics()` - 支持完整的图统计信息
  - `graph_density()` - 支持图密度计算
  - `connectivity_analysis()` - 支持连通性分析

#### ⚡ 性能优化
- 避免重复计算度字典
- 使用 `Counter` 优化度分布统计
- 合并重复代码逻辑，提升可维护性

#### 🐛 Bug 修复
- 修复缺失的 `networkx` 模块导入
- 修复整数除法导致的精度丢失问题
- 修复连通性分析中的重复代码

#### 📚 文档改进
- 为所有主要模型和算法添加学术参考文献
  - 4个传播模型（LT、IC、SIR、SI）
  - 3个影响力最大化算法（Degree Discount、RIS、IMM）
  - 1个关键工具函数（infection_threshold）
- 新增文档：
  - `CHANGELOG.md` - 完整的更新日志
  - `REFERENCES_DOCUMENTATION.md` - 参考文献文档

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

PyNetIM 采用 **“Python 逻辑层 + C++ 计算层”** 的设计：

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
├─ diffusion_model.h              # 扩散模型公共接口
├─ ic_bind.cpp                    # IC 模型 pybind11 绑定
├─ lt_bind.cpp                    # LT 模型 pybind11 绑定
├─ independent_cascade_model.pyi  # IC Python 类型存根
├─ linear_threshold_model.pyi     # LT Python 类型存根
└─ __init__.py
```

说明：

* `.cpp` 文件仅包含 **Python 绑定逻辑**
* `.pyi` 用于：

  * IDE 自动补全
  * 类型检查（MyPy / PyCharm）

支持的模型：

* Independent Cascade (IC)
* Linear Threshold (LT)

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
* 为 IC / LT 等模型提供高效邻接访问

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

说明：

* Python 版本 **逻辑完整**
* 用于：

  * 算法理解
  * 实验修改
* `run_monte_carlo_diffusion.py` 提供统一的蒙特卡洛扩散接口

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
- **GLM-4.7 (Trae AI Assistant)** - 在项目开发过程中提供了关键的技术支持，特别是在 C++/Python 互操作、对象生命周期管理等复杂问题上提供了精准的分析和解决方案

---

## 📄 License

MIT License
详见 [LICENSE](LICENSE)
