# 阶跃函数
def step_function0(x):
    if x > 0:
        return 1
    else:
        return 0
    
import numpy as np

def step_function(x):
    return np.array(x > 0, dtype=int)

# sigmoid函数

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax0(x):
    return np.exp(x) / np.sum(np.exp(x))

def softmax(x):
    # 为了防止溢出，减去最大值
    # 如果是二维矩阵
    if x.ndim == 2:
        y = np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)
    # 如果是一维矩阵
    else:
        y = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    return y

def identity(x):
    return x

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

if __name__ == '__main__':
    x = np.array([-1.0, 1.0, 2.0])
    print(step_function(x))
    print(sigmoid(x))
    print(tanh(x))
    print(relu(x))


    X = np.array([[0,1, 2], [3,4, 5]])
    print(softmax(X))
