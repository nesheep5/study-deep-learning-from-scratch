import numpy as np

class Affine:
    # W:重み b:バイアス
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T) # Tは転置(行列入れ替え)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

# 以下動作確認
W = np.array([[1, 1],[1, 1], [1, 1]])
b = np.array([1, 2])
affine_layer = Affine(W, b)

x = np.array([[1, 2, 3],[4, 5, 6]])
forward = affine_layer.forward(x)
print("forward:\n", forward)