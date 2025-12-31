# PyNetIM

[PyNetIM](https://zzzkhj.github.io/PyNetIM/) 是一个用于**社交网络影响力最大化（Influence Maximization, IM）**问题的 Python 库，集成了多种经典算法与扩散模型，并提供 **Python 实现 + C++ 加速后端**，适用于算法复现、性能对比与科研实验。

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

## 🚀 快速开始

```python
import networkx as nx
from pynetim.py import IMGraph
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

## 📄 License

MIT License
详见 [LICENSE](LICENSE)
