# 最適化

# SGD(確率的勾配降下法)
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr # 学習計数

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
