# ChainerのDroput実装例

class Dropout:
    def __init__(self, dropout_ratio+0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(seld, x, train_flg=True):
        if train_flg:
            self.mask = np.rondom.rand(*x.shape) > self.dropout_ratio
            retun x * self.mask

        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask