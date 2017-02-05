# シグモイドレイヤ
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

# 以下動作確認
import numpy as np
x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print("x:\n",  x)
sigmoid=layer = Sigmoid()
forward = sigmoid=layer.forward(x)
print("forward:\n", forward)

dx = np.array([[1,1,],[1,1]])
print("dx:\n",  dx)
backward = sigmoid=layer.backward(dx)
print("backward:\n", backward)
