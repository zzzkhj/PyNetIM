# 更新日志 (Changelog)

本文档记录 PyNetIM 项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/)。

---

## 版本历史

| 版本 | 发布日期 | 主要更新 |
|------|----------|----------|
| [v0.5.2](changelog/v0.5.2.md) | 2026-04-16 | 深度学习算法模块、训练框架、pybind 签名文档修复 |
| [v0.5.1](changelog/v0.5.1.md) | 2026-04-10 | 评估指标模块、时间测量工具、7个启发式算法 |
| [v0.5.0](changelog/v0.5.0.md) | 2026-04-07 | 模块扁平化、算法模块、自定义传播模型、OPIM算法、中文输出 |
| [v0.4.5](changelog/v0.4.5.md) | 2026-04-04 | SI/SIR 扩散模型、统一权重支持 |
| [v0.4.4](changelog/v0.4.4.md) | 2026-03-30 | 激活频率记录、随机种子改进 |
| [v0.4.3](changelog/v0.4.3.md) | 2026-03-30 | 激活节点记录功能 |
| [v0.4.2](changelog/v0.4.2.md) | 2026-03-24 | 线程安全修复 |
| [v0.4.1](changelog/v0.4.1.md) | 2026-03-24 | 性能优化 |
| [v0.4.0](changelog/v0.4.0.md) | 2026-03-23 | C++ 图支持、学术参考文献 |
| [v0.3.0](changelog/v0.3.0.md) | 2025-XX-XX | 初始版本发布 |

---

## 最新版本

### [v0.5.2] - 2026-04-16

**新增功能**:
- 深度学习影响力最大化算法模块 (`pynetim.algorithms.deep_learning`)
  - ToupleGDD: 三重门控图神经网络 + DQN (IEEE TCSS 2024)
  - S2V-DQN: Structure2Vec + DQN (NeurIPS 2017)
  - BiGDN: 端到端图神经网络 + DQN (Expert Syst. Appl. 2025)
  - BiGDNS: BiGDN 学生模型，支持知识蒸馏
- 训练框架: ToupleGDDTrainer, S2VDQNTrainer, BiGDNTrainer, BiGDNNodeEncoderTrainer
- 支持 topk 一次性选择和迭代选择两种推理模式
- 预训练权重自动加载

**Bug 修复**:
- 修复 pybind 算法绑定文件签名文档重复问题（IMM/TIM/OPIM/BaseRIS）
- 修复 opim_algorithm.pyi 中 `run()` 方法多余的 `mode` 参数
- 添加 SI/SIR 模型缺失的 `set_beta()`, `set_max_steps()`, `set_gamma()` 方法绑定
- 添加 graph.pyi 缺失的 `get_adj_list_py()` 方法

**改进**:
- 所有 pybind 绑定添加友好的中文错误提示，替代冗长的默认错误信息

👉 [查看完整更新内容](changelog/v0.5.2.md)

### [v0.5.1] - 2026-04-10

**新增功能**:
- 评估指标模块 (evaluation)：排名指标、影响力指标、种子质量指标、网络指标
- 时间测量模块 (timing)：装饰器、AlgorithmTimer、多次运行统计
- 工具函数 (utils)：最短路径计算（支持跳数和权重）
- 7个经典启发式算法：DegreeCentrality, PageRank, VoteRank, K-shell, Betweenness, Closeness, Eigenvector

**API 变更**:
- 参数重命名：`rounds` → `mc_rounds`，`seed` → `random_seed`
- 所有 `random_seed` 默认值改为 `None`（每次结果不同）
- 多线程参数验证：`use_multithread=True` 时 `num_threads` 必须 > 0

👉 [查看完整更新内容](changelog/v0.5.1.md)

### [v0.5.0] - 2026-04-07

**重大变更**:
- 模块结构扁平化，C++ 模块直接作为主模块
- `pynetim.py` 模块进入维护模式
- `out_neighbors()` 返回类型变更

**新增功能**:
- 影响力最大化算法模块 (`pynetim.algorithms`)
  - 启发式: SingleDiscount, DegreeDiscount
  - 模拟类: Greedy, CELF, CELF++
  - RIS类: BaseRIS, IMM, TIM, TIM+
  - OPIM类: OPIM, OPIM-C (SIGMOD 2018)
- 自定义传播模型支持 (`BaseCallbackDiffusionModel`, `BaseMultiprocessDiffusionModel`)
- 类型提示改进，`help()` 显示清晰类型
- 所有输出信息本地化为中文

👉 [查看完整更新内容](changelog/v0.5.0.md)

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

**最后更新**: 2026-04-16
