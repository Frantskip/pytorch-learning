import numpy as np
import os
import sys
# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.layers import *
from common.functions import softmax, sigmoid, cross_entropy_error, numerical_gradient
from collections import OrderedDict
class TwoLayerNet:
    # 初始化
    def __init__(self, input_size, hidden_size, output_size,weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)
        # 定义层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # 输出层
        self.lastLayer = SoftmaxWithLoss()

    # 前向传播
    def forward(self, X):
        # 对于神经网络的每一层，依次调用forward方法
        for layer in self.layers.values():
            X = layer.forward(X)
        return X
    # 损失函数
    def loss(self, X, t):
        y = self.forward(X)
        return self.lastLayer.forward(y, t)
    
    # 计算准确度
    def accuracy(self, X, t):
        y_proba = self.forward(X)
        y = np.argmax(y_proba, axis=1)
        # 如果 t 是一维数组（原始标签），直接使用
        if t.ndim == 1:
            pass
        # 如果 t 是二维数组（one-hot 编码），转换为原始标签
        else:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(X.shape[0])
        return accuracy
    # 计算梯度
    def numerical_gradient(self, X, t):
        loss_f = lambda W: self.loss(X, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_f, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_f, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_f, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_f, self.params['b2'])
        return grads

    def gradient(self, X, t):
        # 前向传播
        self.loss(X, t)
        # 反向传播
        dy = 1
        dy = self.lastLayer.backward(dy)
        # 反向传播到每一层
        for layer in reversed(self.layers.values()):
            dy = layer.backward(dy)
        # 收集所有层的梯度
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads
if __name__ == '__main__':
    # 1.定义数据
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])
    # 2.定义模型
    net = TwoLayerNet(input_size=2, hidden_size=3, output_size=3)
    # 3.计算梯度
    grads = net.numerical_gradient(x, t)
    print(grads)