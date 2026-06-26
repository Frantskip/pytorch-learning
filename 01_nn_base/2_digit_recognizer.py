import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from common.functions import sigmoid, softmax

# 读取数据
def get_data():
    # 1.从文件读取数据
    data = pd.read_csv(r'D:\桌面\pytorch\train.csv')
    # 2.划分数据集
    X = data.drop(columns=['label'])
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # 3.数据归一化
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return  X_test, y_test

# 初始化网络
def init_network():
    # 直接从文件中加载字典对象
    network = joblib

# 主流程
x , y = get_data()

