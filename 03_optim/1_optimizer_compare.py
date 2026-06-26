import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import sys
# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.optimizer import *

# 定义目标函数：f(x , y) = 1/20 * x^2 + y^2
def f(x, y):
    return 1/20 * x**2 + y**2

# 定义梯度计算方法，得到一个长度为2的向量
def f_grad(x, y):
    return x/10, 2*y

# 定义初始点位置
init_pos = (-7.0, 2.0)

# 定义参数和梯度
params = {}
grads = {}

# 定义优化器列表,指定学习率
optimizers = OrderedDict()
optimizers['SGD'] = SGD(lr=0.1)
optimizers['Momentum'] = Momentum(lr=0.1)
optimizers['AdaGrad'] = AdaGrad(lr=0.1)
optimizers['Adam'] = Adam(lr=0.1)

idx = 1

# 迭代优化器列表
for key in optimizers.keys():
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    # 初始化参数
    params['x'], params['y'] = init_pos[0], init_pos[1]
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        grads['x'], grads['y'] = f_grad(params['x'], params['y'])
        optimizer.update(params, grads)

    # 画图
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    Z[Z > 7] = 0
    # 定义子图
    plt.subplot(2, 2, idx)
    idx += 1
    # 绘制等高线图
    plt.contour(X, Y, Z, levels=10)
    # 单独画出最小值点
    plt.plot(0, 0, '+')
    # 画出点的轨迹
    plt.plot(x_history, y_history,'o-',color='red',markersize=2,label=key)
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    plt.legend()
plt.show()
