from functions import sigmoid, softmax, cross_entropy_error
import numpy as np

# ReLU
class ReLU:
    # 初始化
    def __init__(self):
        self.mask = None

    # 前向传播
    def forward(self, x):
        self.mask = (x <= 0)
        y = x.copy()
        y[self.mask] = 0
        return y
    # 反向传播
    def backward(self, dy):
        dx = dy.copy()
        dx[self.mask] = 0
        return dx

# Sigmoid
class Sigmoid:
    # 初始化
    def __init__(self,x):
        # 定义内部属性，记录输出值y，用于反向传播
        y = sigmoid(x)
        self.y = y
        return y
    # 反向传播
    def backward(self, dy):
        dx = dy * (1.0 - self.y) * self.y
        return dx
        
# Affine仿射层
class Affine:
    # 初始化
    def __init__(self, W, b):
        self.W = W
        self.b = b
        # 对输入数据X做保存，方便反向传播时使用
        self.X = None
        self.original_x_shape = None
        # 将权重和偏置参数的梯度（偏导数）保存成属性
        self.dW = None
        self.db = None
    # 前向传播
    def forward(self, X):
        self.original_x_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)
        return np.dot(X, self.W) + self.b
    # 反向传播
    def backward(self, dy):
        dX = np.dot(dy, self.W.T)
        dX = dX.reshape(*self.original_x_shape)
        self.dW = np.dot(self.X.T, dy)
        self.db = np.sum(dy, axis=0)
        return dX
    
# 输出层
class SoftmaxWithLoss:
    # 初始化
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    # 前向传播
    def forward(self, X, t):
        self.t = t
        self.y = softmax(X)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    # 反向传播
    def backward(self, dy=1):
        n = self.t.shape[0]
        # 如果是独热编码的标签，就直接带入公式计算
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / n
        else:
            dx = self.y.copy()
            dx[np.arange(n), self.t] -= 1
            dx = dx / n
        return dx