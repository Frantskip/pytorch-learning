# PyTorch Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🧠 从零开始学习深度学习 — 使用 NumPy 手写神经网络，逐步过渡到 PyTorch。
>
> 🧠 Deep learning from scratch — implementing neural networks with NumPy, then transitioning to PyTorch.

---

## 📖 目录 / Table of Contents

- [项目简介 / Introduction](#-项目简介--introduction)
- [项目结构 / Project Structure](#-项目结构--project-structure)
- [环境要求 / Requirements](#-环境要求--requirements)
- [模块说明 / Module Details](#-模块说明--module-details)
- [快速开始 / Quick Start](#-快速开始--quick-start)
- [学习路线 / Learning Path](#-学习路线--learning-path)

---

## 📝 项目简介 / Introduction

**中文**：本项目是一个深度学习入门实践仓库，从最基础的神经网络原理出发，使用 **纯 NumPy** 手工实现神经网络的每一层、每一种激活函数、每一个优化器，深刻理解反向传播和梯度下降的本质。之后再平滑过渡到 **PyTorch** 框架，学习 Tensor 的基本操作。

**English**: This repository is a hands-on deep learning tutorial. Starting from first principles, we implement every neural network layer, activation function, and optimizer using **pure NumPy** — to deeply understand backpropagation and gradient descent. We then smoothly transition to **PyTorch** for tensor operations.

---

## 📁 项目结构 / Project Structure

```
pytorch-learning/
├── 01_nn_base/                  # 神经网络基础 / Neural Network Basics
│   ├── 1_simple_network.py      #   简单3层网络前向传播
│   ├── 2_digit_recognizer.py    #   手写数字识别（数据加载）
│   └── functions.py             #   激活函数实现（副本）
│
├── 02_train/                    # 网络训练 / Network Training
│   ├── 01_simple_net_grad.py    #   数值梯度计算
│   ├── 02_gradtient_desent.py   #   梯度下降算法
│   ├── two_layer_net.py         #   两层神经网络（完整实现）
│   ├── 4_digit_recognizer_nn_train.py  # 手写数字识别训练
│   ├── functions.py             #   激活函数 + 数值梯度
│   └── load_data.py             #   MNIST 数据加载器
│
├── 03_optim/                    # 优化器对比 / Optimizer Comparison
│   └── 1_optimizer_compare.py   #   SGD vs Momentum vs AdaGrad vs Adam
│
├── 04_pytorch_base/             # PyTorch 基础 / PyTorch Basics
│   ├── 1_tensor_creator.ipynb   #   Tensor 创建方式
│   ├── 2_tensor_conversion.ipynb #  Tensor 类型转换
│   └── 3_tensor_calculation.ipynb # Tensor 数学运算
│
├── common/                      # 公共模块 / Common Modules
│   ├── functions.py             #   激活函数 & 损失函数 & 数值梯度
│   ├── layers.py                #   网络层（ReLU, Sigmoid, Affine, SoftmaxWithLoss）
│   ├── optimizer.py             #   优化器（SGD, Momentum, AdaGrad, RMSProp, Adam）
│   ├── load_data.py             #   数据加载工具
│   └── __init__.py              #   包初始化
│
└── README.md
```

---

## 🔧 环境要求 / Requirements

| 依赖 / Dependency | 版本 / Version | 用途 / Purpose |
|-------------------|---------------|----------------|
| Python            | ≥ 3.8         | 运行环境 |
| NumPy             | any           | 矩阵运算、数学计算 |
| Matplotlib        | any           | 优化器轨迹可视化 |
| Pandas            | any           | CSV 数据读取 |
| Scikit-learn      | any           | 数据划分、归一化 |
| PyTorch           | ≥ 2.0         | `04_pytorch_base` 模块 |
| Jupyter Notebook  | any           | `.ipynb` 文件运行 |

安装 / Install：

```bash
pip install numpy matplotlib pandas scikit-learn torch jupyter
```

---

## 📚 模块说明 / Module Details

### 01 — 神经网络基础 / Neural Network Basics

**中文**：从零搭建一个 3 层全连接网络，手动定义权重和偏置，使用 Sigmoid 激活函数，完成一次完整的前向传播，直观感受数据在网络中的流动。

**English**: Build a 3-layer fully-connected network from scratch. Weights and biases are defined manually. Uses Sigmoid activation. Completes one full forward pass to intuitively understand data flow through a network.

```python
# 核心：前向传播 / Core: Forward Pass
a1 = np.dot(x, W1) + b1
z1 = sigmoid(a1)
a2 = np.dot(z1, W2) + b2
z2 = sigmoid(a2)
a3 = np.dot(z2, W3) + b3
y = identity(a3)
```

### 02 — 网络训练 / Network Training

**中文**：实现数值梯度计算、梯度下降优化，以及一个完整的 `TwoLayerNet` 类（Affine → ReLU → Affine → Softmax）。包含前向传播、损失计算、反向传播和准确率评估。

**English**: Implements numerical gradient computation, gradient descent, and a complete `TwoLayerNet` class (Affine → ReLU → Affine → Softmax). Includes forward pass, loss calculation, backpropagation, and accuracy evaluation.

### 03 — 优化器对比 / Optimizer Comparison

**中文**：在同一目标函数 \( f(x, y) = \frac{1}{20}x^2 + y^2 \) 上对比四种优化器的收敛轨迹：

| 优化器 | 特点 |
|--------|------|
| **SGD** | 朴素随机梯度下降，收敛慢，易震荡 |
| **Momentum** | 引入动量项，加速收敛，减少震荡 |
| **AdaGrad** | 自适应学习率，适合稀疏特征 |
| **Adam** | 结合 Momentum 和 AdaGrad，最常用的优化器 |

**English**: Compares convergence trajectories of four optimizers on \( f(x, y) = \frac{1}{20}x^2 + y^2 \):

| Optimizer | Feature |
|-----------|---------|
| **SGD** | Vanilla gradient descent, slow convergence |
| **Momentum** | Momentum term for faster, smoother convergence |
| **AdaGrad** | Adaptive learning rates, suits sparse features |
| **Adam** | Combines Momentum + AdaGrad, the go-to optimizer |

![Optimizer Comparison](docs/optimizer_comparison.png)

### 04 — PyTorch 基础 / PyTorch Basics

**中文**：三个 Jupyter Notebook 覆盖 PyTorch 入门核心操作：

1. **Tensor 创建**：`torch.tensor()`, `torch.zeros()`, `torch.randn()` 等
2. **Tensor 转换**：CPU ↔ GPU, dtype 转换, NumPy ↔ Tensor
3. **Tensor 运算**：矩阵乘法、广播、索引、自动微分初探

**English**: Three Jupyter Notebooks covering essential PyTorch operations:

1. **Tensor Creation**: `torch.tensor()`, `torch.zeros()`, `torch.randn()`, etc.
2. **Tensor Conversion**: CPU ↔ GPU, dtype casting, NumPy ↔ Tensor
3. **Tensor Calculation**: Matrix multiplication, broadcasting, indexing, autograd basics

### common/ — 公共模块 / Common Modules

**中文**：本项目核心 — 纯 NumPy 手工实现的深度学习组件：

| 文件 | 内容 |
|------|------|
| `functions.py` | `sigmoid`, `softmax`, `relu`, `tanh`, `cross_entropy_error`, `numerical_gradient` |
| `layers.py` | `ReLU`, `Sigmoid`, `Affine`, `SoftmaxWithLoss` — 每层含 forward + backward |
| `optimizer.py` | `SGD`, `Momentum`, `AdaGrad`, `RMSProp`, `Adam` — 完整的 update 逻辑 |
| `load_data.py` | MNIST 风格数据加载 |

**English**: The heart of this project — deep learning components implemented entirely in NumPy:

| File | Contents |
|------|----------|
| `functions.py` | `sigmoid`, `softmax`, `relu`, `tanh`, `cross_entropy_error`, `numerical_gradient` |
| `layers.py` | `ReLU`, `Sigmoid`, `Affine`, `SoftmaxWithLoss` — each with forward + backward |
| `optimizer.py` | `SGD`, `Momentum`, `AdaGrad`, `RMSProp`, `Adam` — complete update logic |
| `load_data.py` | MNIST-style data loader |

---

## 🚀 快速开始 / Quick Start

### 运行简单前向传播 / Run Simple Forward Pass

```bash
python 01_nn_base/1_simple_network.py
```

### 训练两层网络 / Train a Two-Layer Net

```bash
python 02_train/two_layer_net.py
```

### 对比优化器 / Compare Optimizers

```bash
python 03_optim/1_optimizer_compare.py
```

### 打开 PyTorch Notebook / Open PyTorch Notebook

```bash
jupyter notebook 04_pytorch_base/
```

---

## 🗺️ 学习路线 / Learning Path

```
┌─────────────────────────────────────────────────┐
│  ① 前向传播 / Forward Pass                       │
│     NumPy 实现 3 层网络，理解数据流动              │
│     → 01_nn_base/1_simple_network.py             │
├─────────────────────────────────────────────────┤
│  ② 损失函数 & 激活函数                            │
│     Sigmoid, Softmax, 交叉熵，手写数字识别准备      │
│     → common/functions.py                        │
├─────────────────────────────────────────────────┤
│  ③ 层抽象 / Layer Abstraction                    │
│     ReLU, Affine, SoftmaxWithLoss 类封装           │
│     → common/layers.py                           │
├─────────────────────────────────────────────────┤
│  ④ 反向传播 / Backpropagation                    │
│     TwoLayerNet 完整训练流程，梯度计算              │
│     → 02_train/two_layer_net.py                  │
├─────────────────────────────────────────────────┤
│  ⑤ 优化器 / Optimizers                           │
│     SGD → Momentum → AdaGrad → Adam 可视化对比     │
│     → 03_optim/                                  │
├─────────────────────────────────────────────────┤
│  ⑥ PyTorch 入门                                  │
│     Tensor 创建 → 转换 → 运算 → 自动微分           │
│     → 04_pytorch_base/                           │
└─────────────────────────────────────────────────┘
```

> 💡 **建议**：按照 ① → ⑥ 的顺序学习。先用 NumPy 理解原理，再用 PyTorch 提高效率。
>
> 💡 **Suggestion**: Follow ① → ⑥ in order. Understand the principles with NumPy first, then leverage PyTorch for efficiency.

---

## 📄 License

MIT © [Frantskip](https://github.com/Frantskip)
