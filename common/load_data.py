import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
def get_data():
    # 读取数据
    data = pd.read_csv(r'D:\桌面\pytorch\train.csv')
    # 划分特征和目标变量
    X = data.drop(columns=['label'])
    y = data['label']
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 数据归一化
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 将数据都转换成nparray
    y_train = y_train.values
    y_test = y_test.values
    return X_train, X_test, y_train, y_test