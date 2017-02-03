# 乗算レイヤー
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # 順伝搬
    def forward(self, x, y):
        self.x = x
        self.y = y
        return  x * y

    # 逆伝搬 dout:微分値
    def backward(self, dout):
        #  XとYをひっくり返して乗算
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

# 加算レイヤー
class AddLayer:
    def __init__(self):
        # passは「何も行わない」という命令
        pass

    # 順伝搬
    def forward(self, x, y):
        return x + y

    # 逆伝搬 dout:微分値
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
