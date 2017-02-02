import numpy as np
import matplotlib.pyplot as plt

# 重みとバイアスの初期化
def init_network():
    network = {}
    # 1層目
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    # 2層目
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    # 3層目
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

# 入力→出力
def forword(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 1層目
    a1 = np.dot(x, W1) +b1  # A = XW +B
    z1 = sigmoid(a1)        # Z = h(A)
    # 2層目
    a2 = np.dot(z1, W2) +b2
    z2 = sigmoid(a2)
    # 3層目
    a3 = np.dot(z2, W3) +b3
    y = identity_function(a3)   # 最後の層のみ活性化関数が異なる

    return y

# シグモイド関数(活性化関数)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 恒等関数(活性化関数)
def identity_function(x):
    return x

# 以下動作確認
network = init_network()
x = np.array([1.0, 0.5])
y = forword(network, x)
print(y) # [0.31682708  0.69627909]