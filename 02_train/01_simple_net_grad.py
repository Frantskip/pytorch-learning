import numpy as np
from common.functions import softmax, cross_entropy_error

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad

# 定义一个简单神经网络类
class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 2x3的随机矩阵

    # 前向传播
    def forward(self, X):
        a = X @ self.W
        y = softmax(a)
        return y
     # 损失函数
    def loss(self, x, t):
        y = self.forward(x)
        return cross_entropy_error(y, t)
    

if __name__ == '__main__':  
    # 1.定义数据
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])
    # 2.定义模型
    net = SimpleNet()
    # 3.计算梯度
    f = lambda w: net.loss(x, t)
    gradw = numerical_gradient(f, net.W)
    print(gradw) 