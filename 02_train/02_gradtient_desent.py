import numpy as np
import matplotlib.pyplot as plt
from common.functions import numerical_gradient

# 定义梯度下降法的函数
def gradient_descent(f, init_x, lr=0.01, num_iter=100):
    x = init_x
    # 定义列表保存x的变化
    x_history = []
    for i in range(num_iter):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x , np.array(x_history)

# 定义目标函数
def f(x):
    return x[0]**2+x[1]**2

if __name__ == '__main__':
    # 1.定义初始值
    init_x = np.array([-3.0, 4.0])
    # 2.定义超参数
    lr = 0.1
    num_iter = 20
    # 3.调用梯度下降法
    x, x_history = gradient_descent(f, init_x, lr=lr, num_iter=num_iter)
    # 4.打印结果
    print(x)
    print(x_history)
    # 5.可视化结果
    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.scatter(x_history[:,0], x_history[:,1], c='r')
    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.show()