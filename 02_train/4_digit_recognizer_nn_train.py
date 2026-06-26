import numpy as np
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
from load_data import get_data

# 1.加载数据
X_train, X_test, y_train, y_test = get_data()

# 2.创建模型
net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 3.定义超参数
learning_rate = 0.1
batch_size = 100
num_epochs = 10

train_size = X_train.shape[0]
iter_per_epoch = np.ceil(train_size / batch_size)
iter_num = int(iter_per_epoch * num_epochs)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 4.循环迭代，用梯度下降法训练模型
for i in range(iter_num):
    # 4.1 随机抽取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    X_batch = X_train[batch_mask]
    t_batch = y_train[batch_mask]
    # 4.2 计算梯度
    # grad = net.numerical_gradient(X_batch, t_batch)
    grad = net.gradient(X_batch, t_batch)

    # print("grad:=====================",i)
    # 4.3 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        net.params[key] -= learning_rate * grad[key]
    # 4.4 计算损失函数
    loss = net.loss(X_batch, t_batch)
    train_loss_list.append(loss)
    # 4.5 计算训练集和测试集的准确率
    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(X_train, y_train)
        test_acc = net.accuracy(X_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"epoch: {i//iter_per_epoch}, train_loss: {loss:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")

# 5.绘制损失函数和准确率的变化图
# 绘制损失函数
x_loss = np.arange(len(train_loss_list))
plt.figure(figsize=(12, 5))

# 子图1：损失函数
plt.subplot(1, 2, 1)
plt.plot(x_loss, train_loss_list, label='train_loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend(loc='best')

# 子图2：准确率
plt.subplot(1, 2, 2)
x_acc = np.arange(len(train_acc_list))
plt.plot(x_acc, train_acc_list, label='train_acc')
plt.plot(x_acc, test_acc_list, label='test_acc', linestyle='--')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='best')

plt.tight_layout()
plt.show()