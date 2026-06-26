import numpy as np

# 随机梯度下降
class SGD:
    # 初始化
    def __init__(self, lr=0.01):
        self.lr = lr
    # 更新参数
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# 动量梯度下降
class Momentum:
    # 初始化
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None  # 历史负梯度的加权和
    # 更新参数
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

class AdaGrad:
    # 初始化
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None  # 历史负梯度的平方和
    # 更新参数
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-8)
    
class RMSProp:
    # 初始化
    def __init__(self, lr=0.01, decay_rate=0.9):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None  # 历史负梯度的平方和
    # 更新参数
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] = self.decay_rate * self.h[key] + (1 - self.decay_rate) * grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-8)

class Adam:
    # 初始化
    def __init__(self, lr=0.01, alpha1=0.9, alpha2=0.999):
        self.lr = lr
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.h = None  # 历史负梯度的加权和
        self.v = None  # 历史负梯度的平方和
        self.t = 0  # 迭代次数
    # 更新参数
    def update(self, params, grads):
        if self.v is None:
            self.v, self.h = {}, {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                self.h[key] = np.zeros_like(val)
        self.t += 1 # 迭代次数+1
        lr_t = self.lr * np.sqrt(1.0 - self.alpha2 ** self.t) / (1.0 - self.alpha1 ** self.t)
        for key in params.keys():
            self.v[key] += (1 - self.alpha1) * (grads[key]-self.v[key])
            self.h[key] += (1 - self.alpha2) * (grads[key]**2 - self.h[key])
            params[key] -= lr_t * self.v[key] / (np.sqrt(self.h[key]) + 1e-8)