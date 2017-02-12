import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

# 誤差逆伝搬に対応した２層ニューラルネットワークの実装
class TwoLayerNet:

    # 初期化処理
    # 引数: 入力層のニューロン数、隠れ層のニューロン数、出力層のニューロン数ｍ重み初期化字のガウス分布のスケール
    def __init__(self,input_size, hidden_size, output_size, weight_init_std):
        # 重みの初期化
        self.param = {}
        self.param['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.param['b1'] = np.zeros(hidden_size)
        self.param['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.param['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affin1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affin2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    #  認識(推論)をおこなう
    # 引数 x： 画像データ
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

    # 損失関数の値を求める
    # 引数 x:画像データ、 t:正解ラベル
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    # 認識精度を求める
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 重みパラメータに対する勾配を数値微分によって求める
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self,param['W1'])
        grads['b1'] = numerical_gradient(loss_W, self,param['b1'])
        grads['W2'] = numerical_gradient(loss_W, self,param['W2'])
        grads['b2'] = numerical_gradient(loss_W, self,param['b2'])

        return grads

    #重みパラメータに対する勾配を誤差逆伝搬法によって求める
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.valuses())
        layers = reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
