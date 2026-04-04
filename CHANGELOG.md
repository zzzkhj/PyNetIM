# 更新日志 (Changelog)

本文档记录 PyNetIM 项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/)。

## [0.4.5] - 2026-04-04

### 新增 (Added)
- **SI 和 SIR 扩散模型 (C++ 后端)**:
  - 新增 `SusceptibleInfectedModel` - SI 传播模型
  - 新增 `SusceptibleInfectedRecoveredModel` - SIR 传播模型
  - 支持参数设置：`beta`（感染概率）、`gamma`（恢复概率）、`max_steps`（最大步数）
  - 支持单次模拟和 Monte Carlo 多轮模拟
  - 支持多线程并行模拟
  - 支持激活节点记录和激活频率记录

- **Graph 构造函数支持统一权重**:
  - `IMGraphCpp(num_nodes, edges, weights=1.0)` - 默认所有边权重为 1.0
  - `IMGraphCpp(num_nodes, edges, 0.3)` - 所有边使用统一权重 0.3
  - `IMGraphCpp(num_nodes, edges, [0.1, 0.2, ...])` - 每条边使用各自的权重

### API 更新 (API Changes)
- **SusceptibleInfectedModel**:
  - 构造函数：`SusceptibleInfectedModel(graph, seeds, beta=0.1, max_steps=100, record_activated=False, record_activation_frequency=False)`
  - `set_beta(beta: float)` - 设置感染概率
  - `set_max_steps(max_steps: int)` - 设置最大传播步数
  - `run_single_simulation(seed=None) -> int` - 单次模拟
  - `run_monte_carlo_diffusion(rounds, seed=None, use_multithread=False, num_threads=0) -> float` - Monte Carlo 模拟
  - `get_activated_nodes() -> Set[int]` - 获取激活节点集合
  - `get_activation_frequency() -> List[int]` - 获取激活频率

- **SusceptibleInfectedRecoveredModel**:
  - 构造函数：`SusceptibleInfectedRecoveredModel(graph, seeds, beta=0.1, gamma=0.1, record_activated=False, record_activation_frequency=False)`
  - `set_beta(beta: float)` - 设置感染概率
  - `set_gamma(gamma: float)` - 设置恢复概率
  - `run_single_simulation(seed=None) -> int` - 单次模拟
  - `run_monte_carlo_diffusion(rounds, seed=None, use_multithread=False, num_threads=0) -> float` - Monte Carlo 模拟
  - `get_activated_nodes() -> Set[int]` - 获取激活节点集合
  - `get_activation_frequency() -> List[int]` - 获取激活频率

- **IMGraphCpp**:
  - `weights` 参数现在支持 `float` 类型，表示所有边使用统一权重

### 代码重构 (Refactoring)
- **扩散模型文件分离**:
  - 将扩散模型从单一文件分离到独立文件
  - `independent_cascade.h` - IC 模型
  - `linear_threshold.h` - LT 模型
  - `susceptible_infected.h` - SI 模型
  - `susceptible_infected_recovered.h` - SIR 模型
  - `common.h` - 共享代码（ObjectPool、RNG 工具函数）

### 测试 (Testing)
- 新增完整测试覆盖：
  - `test_ic_model.py` - IC 模型完整测试
  - `test_lt_model.py` - LT 模型完整测试
  - `test_si_model.py` - SI 模型完整测试
  - `test_sir_model.py` - SIR 模型完整测试
- 单线程和多线程结果一致性验证通过
- 参数验证测试通过（beta、gamma 范围检查）

### 技术细节 (Technical Details)
- **修改文件**:
  - `src/pynetim/cpp/include/common.h` - 共享工具代码
  - `src/pynetim/cpp/include/independent_cascade.h` - IC 模型
  - `src/pynetim/cpp/include/linear_threshold.h` - LT 模型
  - `src/pynetim/cpp/include/susceptible_infected.h` - SI 模型
  - `src/pynetim/cpp/include/susceptible_infected_recovered.h` - SIR 模型
  - `src/pynetim/cpp/include/diffusion_model.h` - 主头文件
  - `src/pynetim/cpp/bindings/si_bind.cpp` - SI 模型绑定
  - `src/pynetim/cpp/bindings/sir_bind.cpp` - SIR 模型绑定
  - `src/pynetim/cpp/include/Graph.h` - Graph 类统一权重支持
  - `src/pynetim/cpp/bindings/graph_bind.cpp` - Graph 绑定更新

- **实现细节**:
  - SI 模型：每个时间步，感染节点以概率 beta 尝试感染邻居
  - SIR 模型：在 SI 基础上，感染节点每步以概率 gamma 恢复
  - 参数验证：beta、gamma ∈ (0, 1]，max_steps > 0
  - 多线程种子预生成策略确保单线程和多线程结果完全一致

### 使用示例 (Usage Examples)
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

# 记录激活节点
sir = diffusion_model.SusceptibleInfectedRecoveredModel(
    graph, seeds, beta=0.3, gamma=0.1, record_activated=True
)
count = sir.run_single_simulation(seed=42)
activated = sir.get_activated_nodes()
print(f"激活节点: {activated}")
```

## [0.4.4] - 2026-03-30

### API 更新 (API Changes)
- **IndependentCascadeModel**:
  - `run_single_simulation()` 的返回值从 `Set[int]` 改为 `int`（激活节点数量）
  - 新增 `get_activated_nodes() -> Set[int]` 方法，用于获取记录的激活节点集合
  - 新增 `get_activation_frequency() -> List[int]` 方法，用于获取每个节点的激活频率
  - 新增 `set_record_activation_frequency(record: bool)` 方法，用于动态开启/关闭激活频率记录
  - `run_single_simulation()` 的 `seed` 参数类型从 `int` 改为 `int | None`，默认值从 `0` 改为 `None`
  - `run_monte_carlo_diffusion()` 的 `seed` 参数类型从 `int` 改为 `int | None`，默认值从 `0` 改为 `None`
  - 构造函数新增 `record_activation_frequency` 参数（默认为 `false`）

- **LinearThresholdModel**:
  - `run_single_simulation()` 的返回值从 `Set[int]` 改为 `int`（激活节点数量）
  - 新增 `get_activated_nodes() -> Set[int]` 方法，用于获取记录的激活节点集合
  - 新增 `get_activation_frequency() -> List[int]` 方法，用于获取每个节点的激活频率
  - 新增 `set_record_activation_frequency(record: bool)` 方法，用于动态开启/关闭激活频率记录
  - `run_single_simulation()` 的 `seed` 参数类型从 `int` 改为 `int | None`，默认值从 `0` 改为 `None`
  - `run_monte_carlo_diffusion()` 的 `seed` 参数类型从 `int` 改为 `int | None`，默认值从 `0` 改为 `None`
  - 构造函数新增 `record_activation_frequency` 参数（默认为 `false`）

### 新增 (Added)
- **分离模拟执行和结果获取**:
  - `run_single_simulation()` 现在只返回激活节点数量，不返回节点集合
  - 新增 `get_activated_nodes()` 方法，专门用于获取记录的激活节点集合
  - 只有在 `record_activated=True` 时，`get_activated_nodes()` 才返回有效数据
  - 当 `record_activated=False` 时，`get_activated_nodes()` 返回空集合

- **多轮模拟记录激活节点并集**:
  - `run_monte_carlo_diffusion()` 现在支持记录所有试验的激活节点并集
  - 当 `record_activated=True` 时，`get_activated_nodes()` 返回所有试验中被激活过的节点（并集）
  - 对于单次模拟：返回该次模拟的激活节点集合
  - 对于 Monte Carlo 模拟：返回所有试验的激活节点并集

- **激活频率记录功能**:
  - 新增 `record_activation_frequency` 参数（默认为 `false`），用于控制是否记录每个节点的激活频率
  - 新增 `set_record_activation_frequency(record)` 方法，可在运行时动态开启/关闭记录功能
  - 新增 `get_activation_frequency()` 方法，返回每个节点被激活的次数
  - 只有在 `record_activation_frequency=True` 时，`get_activation_frequency()` 才返回有效数据
  - 当 `record_activation_frequency=False` 时，`get_activation_frequency()` 返回全 0 的列表
  - 可以同时开启 `record_activated` 和 `record_activation_frequency`，记录两种数据

### 优化 (Improved)
- **改进随机种子生成机制**:
  - `run_single_simulation()` 和 `run_monte_carlo_diffusion()` 方法现在都支持 `seed=None` 参数
  - 当 `seed=None` 时，使用多个 `std::random_device` 值的组合来增强随机性
  - 当 `seed=固定值` 时，使用固定种子，确保模拟结果可重现
  - 改进后的随机种子生成方式解决了在某些平台（虚拟环境、容器）上 `std::random_device` 返回相同值的问题
  - 使用 `std::seed_seq` 和多个随机值组合，确保每次调用都能得到不同的结果

- **多线程随机种子一致性**:
  - `run_monte_carlo_diffusion()` 采用种子预生成策略
  - 在开始任何模拟之前，使用主随机数生成器预生成所有试验的种子
  - 每个试验使用独立的种子，确保单线程和多线程结果完全一致
  - 多线程只是改变了试验的执行顺序和并行度，但没有改变每个试验使用的随机数序列

### 修复 (Fixed)
- **修复 `set_record_activated()` 的清空逻辑**:
  - 当 `set_record_activated(False)` 被调用时，现在会清空 `last_activated_nodes`
  - 避免了返回过期的激活节点数据
  - 确保当 `record_activated=False` 时，`get_activated_nodes()` 总是返回空集合

- **修复 `set_record_activation_frequency()` 的清空逻辑**:
  - 当 `set_record_activation_frequency(False)` 被调用时，现在会清空 `activation_frequency`
  - 避免了返回过期的激活频率数据
  - 确保当 `record_activation_frequency=False` 时，`get_activation_frequency()` 总是返回全 0 的列表

### 测试 (Testing)
- **新增功能验证测试**:
  - IC 模型单次模拟测试（简单链式图、星形图、低权重图）
  - LT 模型单次模拟测试
  - 多个种子节点测试
  - `record_activated=False` 测试
  - `set_record_activated()` 动态切换测试
  - 多次调用 `get_activated_nodes()` 测试
  - 无向图测试
  - 随机种子功能测试
  - 多轮模拟测试（单线程和多线程）
  - 大规模模拟测试（10000 轮）
  - `record_activated` 对多轮模拟性能影响测试
  - 多轮模拟 vs 单次模拟对比测试
  - 编译和运行测试全部通过

- **测试结果**:
  - 简单链式图 (权重=1.0): 激活节点数量=5, 激活节点集合={0, 1, 2, 3, 4} ✅
  - 星形图 (权重=1.0): 激活节点数量=5, 激活节点集合={0, 1, 2, 3, 4} ✅
  - 低权重图 (权重=0.01): 激活节点数量=1, 激活节点集合={0} ✅
  - 多个种子节点: 激活节点数量=6, 激活节点集合={0, 1, 2, 3, 4, 5} ✅
  - record_activated=False: 返回激活节点数量，不记录节点集合 ✅
  - record_activated=True: 返回激活节点数量，记录节点集合 ✅
  - 动态切换 record_activated: 正确清空和记录节点集合 ✅
  - 多次调用 get_activated_nodes(): 返回相同的集合 ✅
  - 无向图: 正确处理无向图的传播 ✅
  - 随机种子 (seed=None): 每次模拟结果可能不同 ✅
  - 固定种子 (seed=42): 多次模拟结果完全相同 ✅
  - 单线程和多线程: 多轮模拟结果完全一致（差异 < 1e-10）✅
  - 固定种子可重现性: 3次运行结果完全相同 ✅
  - record_activated 影响: 不影响多轮模拟结果（差异 = 0.0000）✅
  - 大规模模拟: 10000 轮模拟稳定运行 ✅

### 技术细节 (Technical Details)
- **修改文件**:
  - `src/pynetim/cpp/include/diffusion_model.h` - 添加核心功能实现、改进随机种子生成、修复清空逻辑
  - `src/pynetim/cpp/bindings/ic_bind.cpp` - IC 模型 Python 绑定，更新 API 文档
  - `src/pynetim/cpp/bindings/lt_bind.cpp` - LT 模型 Python 绑定，更新 API 文档
  - `src/pynetim/cpp/diffusion_model/independent_cascade_model.pyi` - 类型提示
  - `src/pynetim/cpp/diffusion_model/linear_threshold_model.pyi` - 类型提示

- **实现细节**:
  - `run_single_trial` 方法新增 `activated_nodes` 参数，用于记录激活节点
  - 添加 `last_activated_nodes` 成员变量（`mutable std::set<int>`），存储最后一次模拟的激活节点
  - 当 `record_activated` 为 `true` 时，遍历所有节点收集激活的节点到 `last_activated_nodes`
  - 当 `record_activated` 为 `false` 时，不收集激活节点，`last_activated_nodes` 保持为空
  - `run_single_simulation()` 返回激活节点数量（`int`），不再返回节点集合
  - 新增 `get_activated_nodes()` 方法，返回 `last_activated_nodes` 的副本
  - `set_record_activated()` 方法在设置为 `false` 时清空 `last_activated_nodes`
  - 使用独立的随机数生成器确保每次模拟的独立性
  - 保持向后兼容性，`record_activated` 默认为 `false`
  - 随机种子生成改进：
    ```cpp
    if (use_random_seed) {
        std::random_device rd;
        std::array<unsigned int, 4> seed_data;
        for (auto& item : seed_data) {
            item = rd();
        }
        std::seed_seq seq(seed_data.begin(), seed_data.end());
        rng.seed(seq);
    }
    ```
  - 多线程种子预生成策略：
    ```cpp
    std::vector<unsigned int> trial_seeds(rounds);
    {
        std::mt19937 master_rng;
        if (seed == 0) {
            std::random_device rd;
            std::array<unsigned int, 4> seed_data;
            for (auto& item : seed_data) {
                item = rd();
            }
            std::seed_seq seq(seed_data.begin(), seed_data.end());
            master_rng.seed(seq);
        } else {
            master_rng.seed(seed);
        }
        for (int i = 0; i < rounds; ++i) {
            trial_seeds[i] = master_rng();
        }
    }
    ```
  - Python 绑定层检测 `seed` 参数是否为 `None`，自动转换为 `use_random_seed` 标志
  - `record_activated` 参数不影响 `run_monte_carlo_diffusion()` 的性能和结果
  - 多线程模式下，每个线程使用独立的随机数生成器，确保线程安全和结果一致性

### 使用示例 (Usage Examples)
```python
# IC 模型
from pynetim.cpp.diffusion_model import IndependentCascadeModel

# 不记录激活节点（默认）
ic_model = IndependentCascadeModel(graph, seeds)
count = ic_model.run_single_simulation()  # 随机种子
print(f"激活节点数量: {count}")

# 记录激活节点
ic_model = IndependentCascadeModel(graph, seeds, record_activated=True)
count = ic_model.run_single_simulation(seed=42)  # 固定种子
activated_nodes = ic_model.get_activated_nodes()
print(f"激活节点数量: {count}")
print(f"激活的节点: {activated_nodes}")

# 动态切换记录
ic_model.set_record_activated(True)
count = ic_model.run_single_simulation()
activated_nodes = ic_model.get_activated_nodes()

ic_model.set_record_activated(False)
count = ic_model.run_single_simulation()
activated_nodes = ic_model.get_activated_nodes()  # 空集合

# LT 模型
from pynetim.cpp.diffusion_model import LinearThresholdModel

lt_model = LinearThresholdModel(graph, seeds, theta_l=0.0, theta_h=1.0, record_activated=True)
count = lt_model.run_single_simulation()  # 随机种子
activated_nodes = lt_model.get_activated_nodes()
print(f"激活节点数量: {count}")
print(f"激活的节点: {activated_nodes}")

# 多轮模拟
avg_count = ic_model.run_monte_carlo_diffusion(1000)  # 随机种子
print(f"平均激活节点数: {avg_count:.2f}")

avg_count = ic_model.run_monte_carlo_diffusion(1000, seed=42)  # 固定种子
print(f"平均激活节点数: {avg_count:.2f}")

# 多线程多轮模拟
avg_count = ic_model.run_monte_carlo_diffusion(1000, seed=42, use_multithread=True, num_threads=4)
print(f"平均激活节点数: {avg_count:.2f}")
```

## [0.4.3] - 2026-03-30
### API 更新 (API Changes)
- **IndependentCascadeModel**:
  - 构造函数新增 `record_activated` 参数
  - 新增 `set_record_activated(record: bool)` 方法
  - 新增 `run_single_simulation(seed: int | None = None) -> int` 方法
  - 新增 `get_activated_nodes() -> Set[int]` 方法
  - `run_single_simulation` 的返回值从 `Set[int]` 改为 `int`（激活节点数量）
  - `run_single_simulation` 的 `seed` 参数类型从 `int` 改为 `int | None`，默认值从 `0` 改为 `None`

- **LinearThresholdModel**:
  - 构造函数新增 `record_activated` 参数
  - 新增 `set_record_activated(record: bool)` 方法
  - 新增 `run_single_simulation(seed: int | None = None) -> int` 方法
  - 新增 `get_activated_nodes() -> Set[int]` 方法
  - `run_single_simulation` 的返回值从 `Set[int]` 改为 `int`（激活节点数量）
  - `run_single_simulation` 的 `seed` 参数类型从 `int` 改为 `int | None`，默认值从 `0` 改为 `None`

### 测试 (Testing)
- **新增功能验证测试**:
  - IC 模型单次模拟测试
  - LT 模型单次模拟测试
  - 多次独立模拟测试（使用不同 seed）
  - `set_record_activated` 动态切换测试
  - 随机种子功能测试（使用 NetworkX ER 随机图）
  - 固定种子可重现性测试
  - 多轮模拟测试（单线程和多线程）
  - 大规模模拟测试（10000 轮）
  - `record_activated` 对多轮模拟性能影响测试
  - 多轮模拟 vs 单次模拟对比测试
  - 编译和运行测试全部通过

- **测试结果**:
  - 随机种子 (seed=None)：5次模拟产生 4-5 种不同结果 ✅
  - 固定种子 (seed=42)：5次模拟结果完全相同 ✅
  - record_activated=False：返回激活节点数量，不记录节点集合 ✅
  - record_activated=True：返回激活节点数量，记录节点集合 ✅
  - 单线程和多线程：多轮模拟结果完全一致（差异 < 1e-10）✅
  - 固定种子可重现性：3次运行结果完全相同 ✅
  - record_activated 影响：不影响多轮模拟结果（差异 = 0.0000）✅
  - 大规模模拟：10000 轮模拟稳定运行 ✅

### 技术细节 (Technical Details)
- **修改文件**:
  - `src/pynetim/cpp/include/diffusion_model.h` - 添加核心功能实现和改进随机种子生成
  - `src/pynetim/cpp/bindings/ic_bind.cpp` - IC 模型 Python 绑定
  - `src/pynetim/cpp/bindings/lt_bind.cpp` - LT 模型 Python 绑定
  - `src/pynetim/cpp/diffusion_model/independent_cascade_model.pyi` - 类型提示
  - `src/pynetim/cpp/diffusion_model/linear_threshold_model.pyi` - 类型提示

- **实现细节**:
  - `run_single_trial` 方法新增 `activated_nodes` 参数，用于记录激活节点
  - 添加 `last_activated_nodes` 成员变量（`mutable std::set<int>`），存储最后一次模拟的激活节点
  - 当 `record_activated` 为 `true` 时，遍历所有节点收集激活的节点到 `last_activated_nodes`
  - 当 `record_activated` 为 `false` 时，不收集激活节点，`last_activated_nodes` 保持为空
  - `run_single_simulation()` 返回激活节点数量（`int`），不再返回节点集合
  - 新增 `get_activated_nodes()` 方法，返回 `last_activated_nodes` 的副本
  - 使用独立的随机数生成器确保每次模拟的独立性
  - 保持向后兼容性，`record_activated` 默认为 `false`
  - 随机种子生成改进：
    ```cpp
    if (use_random_seed) {
        std::random_device rd;
        std::array<unsigned int, 4> seed_data;
        for (auto& item : seed_data) {
            item = rd();
        }
        std::seed_seq seq(seed_data.begin(), seed_data.end());
        rng.seed(seq);
    }
    ```
  - Python 绑定层检测 `seed` 参数是否为 `None`，自动转换为 `use_random_seed` 标志
  - `record_activated` 参数不影响 `run_monte_carlo_diffusion()` 的性能和结果
  - 多线程模式下，每个线程使用独立的随机数生成器，确保线程安全和结果一致性

### 使用示例 (Usage Examples)
```python
# IC 模型
from pynetim.cpp.diffusion_model import IndependentCascadeModel

# 不记录激活节点（默认）
ic_model = IndependentCascadeModel(graph, seeds)
count = ic_model.run_single_simulation()  # 随机种子
print(f"激活节点数量: {count}")

# 记录激活节点
ic_model = IndependentCascadeModel(graph, seeds, record_activated=True)
count = ic_model.run_single_simulation(seed=42)  # 固定种子
activated_nodes = ic_model.get_activated_nodes()
print(f"激活节点数量: {count}")
print(f"激活的节点: {activated_nodes}")

# 动态切换记录
ic_model.set_record_activated(True)
count = ic_model.run_single_simulation()
activated_nodes = ic_model.get_activated_nodes()

ic_model.set_record_activated(False)
count = ic_model.run_single_simulation()
activated_nodes = ic_model.get_activated_nodes()  # 空集合

# LT 模型
from pynetim.cpp.diffusion_model import LinearThresholdModel

lt_model = LinearThresholdModel(graph, seeds, theta_l=0.0, theta_h=1.0, record_activated=True)
count = lt_model.run_single_simulation()  # 随机种子
activated_nodes = lt_model.get_activated_nodes()
print(f"激活节点数量: {count}")
print(f"激活的节点: {activated_nodes}")

# 多轮模拟
avg_count = ic_model.run_monte_carlo_diffusion(1000, seed=42)
print(f"平均激活节点数: {avg_count:.2f}")
```

---

## [0.4.2] - 2026-03-24

### 修复 (Fixed)
- **修复 Graph 对象生命周期管理问题**:
  - 修复了 Python 层 Graph 对象被垃圾回收后，C++ 层模型仍持有悬空指针导致的段错误
  - 根本原因：Graph 没有以正确的 `std::shared_ptr` 形式被 Python 管理
  - 解决方案：在 `graph_bind.cpp` 中使用 `std::make_shared` 创建 Graph 对象
  - 现在 Python 和 C++ 共享同一个 `shared_ptr`，引用计数统一管理
  - `py::keep_alive` 真正生效，确保 Graph 对象在模型使用期间不会被回收

### 测试 (Testing)
- **新增生命周期验证测试**:
  - `test_weakref.py` - 使用 weakref 验证 Graph 对象生命周期
  - `test_weakref2.py` - 带 seeds 的生命周期测试
  - 测试结果：Graph 对象在模型使用期间保持存活，不会发生段错误

### 技术细节 (Technical Details)
- **关键知识点**: `py::class_<T, std::shared_ptr<T>>` 只保证 Python 用 shared_ptr 管理对象，但前提是这个对象本来就是 shared_ptr 创建的
- **修改文件**: `src/pynetim/cpp/bindings/graph_bind.cpp` (L18-L28)
- **修改内容**: 将 `py::init<int, ...>()` 改为使用 lambda 函数和 `std::make_shared` 创建 Graph 对象

---

## [0.4.1] - 2026-03-24

### 修复 (Fixed)
- **修复 C++ 扩展的线程安全问题**:
  - 修复 `IndependentCascadeModel` 和 `LinearThresholdModel` 在多线程模式下的线程安全问题
  - 通过参数传递替代直接访问成员变量 `seeds`，消除跨线程共享 `std::set` 的状态
  - 解决了与 PyTorch 等多线程库结合使用时的 segmentation fault 问题
  - 经过严格测试：5000 次迭代、8 线程并发、10 图并发均稳定运行
  - 单线程和多线程结果完全一致，确保正确性

### 测试 (Testing)
- **新增严格验证测试**:
  - 单线程/多线程一致性测试
  - 长时间多线程压力测试 (2000 次迭代)
  - 多线程并发访问测试 (8 线程)
  - 多图并发处理测试 (10 个图)
  - 极限压力测试 (5000 次迭代)

---

## [0.4.0] - 2026-03-23

### 新增 (Added)
- **C++ 图支持**: 为所有工具函数添加了对 `IMGraphCpp`（C++ 图）的完整支持
  - `set_edge_weight()` - 支持 C++ 图的边权重设置
  - `infection_threshold()` - 支持 C++ 图的感染阈值计算
  - `graph_statistics()` - 支持 C++ 图的统计信息计算
  - `graph_density()` - 支持 C++ 图的密度计算
  - `connectivity_analysis()` - 支持 C++ 图的连通性分析

### 优化 (Improved)
- **性能优化**:
  - `infection_threshold()` - 避免重复计算度字典
  - `graph_statistics()` - 使用 `Counter` 替代手动循环统计度分布
  - `connectivity_analysis()` - 合并有向图和无向图的分支逻辑，消除重复代码
  - `graph_density()` - 修复整数除法问题，确保返回浮点数

- **代码质量**:
  - 删除无用的 `TYPE_CHECKING` 代码块
  - 统一函数签名，支持多种图类型
  - 优化导入语句，添加缺失的 `networkx` 模块

### 修复 (Fixed)
- **Bug修复**:
  - 修复 `utils.py` 中缺失的 `import networkx as nx` 导入
  - 修复 `graph_density()` 函数中的整数除法 `// 2` 导致的精度丢失
  - 修复 `infection_threshold()` 函数中重复调用 `dict(graph.degree())` 的性能问题
  - **修复 C++ 扩展的线程安全问题**:
    - 修复 `IndependentCascadeModel` 和 `LinearThresholdModel` 在多线程模式下的线程安全问题
    - 通过参数传递替代直接访问成员变量 `seeds`，消除跨线程共享 `std::set` 的状态
    - 解决了与 PyTorch 等多线程库结合使用时的 segmentation fault 问题
    - 经过严格测试：5000 次迭代、8 线程并发、10 图并发均稳定运行
    - 单线程和多线程结果完全一致，确保正确性

### 文档 (Documentation)
- **学术参考文献**: 为所有主要模型和算法添加了学术参考文献
  - **传播模型**:
    - Linear Threshold Model - Granovetter (1978)
    - Independent Cascade Model - Kempe et al. (2003)
    - SIR Model - Kermack & McKendrick (1927)
    - SI Model - Kermack & McKendrick (1927)
  - **影响力最大化算法**:
    - Degree Discount Algorithm - Chen et al. (2009)
    - Reverse Influence Sampling (RIS) - Borgs et al. (2014)
    - IMM Algorithm - Tang et al. (2015)
  - **工具函数**:
    - `infection_threshold()` - Pastor-Satorras & Vespignani (2001, 2015)

- **新增文档**:
  - `UTILS_OPTIMIZATION_SUMMARY.md` - utils.py 优化总结
  - `OPTIMIZATION_SUMMARY.md` - C++ 模块优化总结
  - `REFERENCES_DOCUMENTATION.md` - 参考文献添加文档
  - `CHANGELOG.md` - 本更新日志

### 兼容性 (Compatibility)
- **向后兼容**: 所有修改都保持了向后兼容性
  - NetworkX 图的使用方式完全不变
  - 新增的 C++ 图支持是可选的
  - 函数签名使用 `Union` 类型，支持两种图类型

### 技术细节 (Technical Details)
- **图类型检测**: 使用 `hasattr(graph, 'num_nodes') and hasattr(graph, 'directed')` 检测 C++ 图
- **类型注解**: 使用 `TYPE_CHECKING` 避免循环导入 `IMGraphCpp`
- **统一接口**: NetworkX 图和 C++ 图使用相同的函数接口

---

## [0.3.0] - 2025-XX-XX

### 新增 (Added)
- 初始版本发布
- 实现了多种影响力最大化算法
- 实现了多种传播模型
- 提供了 Python 和 C++ 双实现

---

## 版本说明

### 版本号规范
PyNetIM 遵循 [语义化版本控制](https://semver.org/) (Semantic Versioning)：

- **主版本号 (MAJOR)**: 不兼容的 API 修改
- **次版本号 (MINOR)**: 向下兼容的功能性新增
- **修订号 (PATCH)**: 向下兼容的问题修正

### 版本类型
- **主版本更新**: 重大架构变更、API 不兼容
- **次版本更新**: 新增功能、性能优化
- **修订版本更新**: Bug 修复、文档更新

---

## 更新分类

### 🎯 功能新增
- 新的算法实现
- 新的传播模型
- 新的图操作功能

### ⚡ 性能优化
- 算法效率提升
- 内存使用优化
- 计算速度提升

### 🐛 Bug 修复
- 代码错误修复
- 边界情况处理
- 兼容性问题修复

### 📚 文档改进
- API 文档更新
- 示例代码更新
- 学术参考文献添加

### 🔧 内部改进
- 代码重构
- 测试覆盖提升
- 构建流程优化

---

## 贡献指南

如果您想为 PyNetIM 做出贡献，请：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 联系方式

- **作者**: Zhang Kaijing
- **项目主页**: https://zzzkhj.github.io/PyNetIM/
- **问题反馈**: [GitHub Issues](https://github.com/zzzkhj/PyNetIM/issues)

---

**最后更新**: 2026-03-24