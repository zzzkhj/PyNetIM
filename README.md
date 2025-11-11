# PyNetIM

PyNetIM 是一个用于社交网络中影响力最大化的 Python 库，实现了多种经典算法和扩散模型。

## 功能特点

- 实现了多种影响力最大化算法
- 支持独立级联(IC)和线性阈值(LT)扩散模型，可以自定义扩展（继承BaseDiffusionModel）
- 提供图结构封装和边权重设置工具
- 支持单进程和多进程模式
- 包含算法性能计时器装饰器

## 算法实现

### 启发式算法
- [SingleDiscountAlgorithm](src\pynetim\algorithms\heuristic_algorithm.py#L9-L62): 简单度折扣算法
- [DegreeDiscountAlgorithm](src\pynetim\algorithms\heuristic_algorithm.py#L65-L121): 度折扣算法

### 模拟算法
- [GreedyAlgorithm](src\pynetim\algorithms\simulation_algorithm.py#L10-L87): 贪婪算法
- [CELFAlgorithm](src\pynetim\algorithms\simulation_algorithm.py#L90-L174): CELF算法(Cost-Effective Lazy Forward)

### 反向影响采样算法
- [BaseRISAlgorithm](src\pynetim\algorithms\RIS_algorithm.py#L15-L238): 基础RIS算法
- [IMMAlgorithm](src\pynetim\algorithms\RIS_algorithm.py#L241-L359): IMM算法(Influence Maximization via Martingales)

## 扩散模型

- [IndependentCascadeModel](src\pynetim\diffusion_model\independent_cascade_model.py#L7-L113): 独立级联模型(IC)
- [LinearThresholdModel](src\pynetim\diffusion_model\linear_threshold_model.py#L7-L117): 线性阈值模型(LT)

## 核心组件

### 图结构
- [IMGraph](src\pynetim\graph\graph.py#L7-L133): 影响力最大化图类，封装NetworkX图对象

### 工具函数
- [set_edge_weight](src\pynetim\utils\utils.py#L8-L44): 设置边权重
- [run_monte_carlo_diffusion](src\pynetim\diffusion_model\run_monte_carlo_diffusion.py#L0-L72): 执行蒙特卡洛模拟扩散
- [Timer](src\pynetim\decorator\decorator.py#L4-L72): 计时器装饰器

## 安装

```bash
pip install pynetim
```


## 快速开始

```python
import networkx as nx
from pynetim import IMGraph, IndependentCascadeModel
from pynetim.algorithms import DegreeDiscountAlgorithm

# 创建图对象
g = nx.erdos_renyi_graph(100, 0.1)
# 入度分之一作为权重
graph = IMGraph(g, edge_weight_type='WC')

# 使用算法
algorithm = DegreeDiscountAlgorithm(graph)
seeds = algorithm.run(k=10)
```


## 项目结构

```
pynetim/
├── algorithms/         # 算法实现
├── diffusion_model/    # 扩散模型
├── graph/             # 图结构封装
├── utils/             # 工具函数
├── decorator/         # 装饰器
└── __init__.py        # 包初始化文件
```


## 版本

当前版本: 0.1.0

## 许可证
[MIT](LICENSE)
