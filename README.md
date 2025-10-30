# Cliff Walk环境下SARSA与Q-learning算法对比
通过Cliff Walk环境对比分析SARSA和Q-learning两种经典强化学习算法的性能差异，重点评估：
- 两种算法在高风险环境中的学习效率和稳定性
- 在线学习性能与安全性的权衡关系
- 算法收敛速度和最终策略质量
- 探索策略对风险规避的影响机制

## 项目架构
```txt
cliff_walk_experiment/
├── SARSQ.py              # SARSA算法实现
├── q_learning.py         # Q-learning算法实现  
├── comparison.py         # 对比实验和分析
└── output/               # 结果输出
    ├── SARSA/
    ├── Q-learning/
    └── comparison/
```
## 依赖库
```bash
# 核心依赖
pip install numpy
pip install matplotlib
pip install seaborn
pip install tqdm

# 可选：
pip install scienceplots
```
