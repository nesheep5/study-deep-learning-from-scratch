#ReLUレイヤ
class ReLu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

# 以下動作確認
import numpy as np
x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print("x:\n",  x)
relu_layer = ReLu()
forward = relu_layer.forward(x)
print("forward:\n", forward)

dx = np.array([[1,1,],[1,1]])
print("dx:\n",  dx)
backward = relu_layer.backward(dx)
print("backward:\n", backward)
