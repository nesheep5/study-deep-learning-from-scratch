import numpy as np

#ソフトマックス関数＆損失関数（交差エントロピー誤差）
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None #損失
        self.y = None #softmaxの出力
        self.t = None # 教師データ(one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

    # ソフトマックス関数
    def softmax(a):
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    # 交差エントロピー誤差
    def cross_entropy_error(y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))

