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
        # TODO 未実装

    # 損失関数の値を求める
    # 引数 x:画像データ、 t:正解ラベル
    def loss(self, x, t):
        # TODO 未実装

    # 認識精度を求める
    def accuracy(self, x, t):
        # TODO 未実装

    # 重みパラメータに対する勾配を数値微分によって求める
    def numerical_gradient(self, x, t):
        # TODO 未実装

    #重みパラメータに対する勾配を誤差逆伝搬法によって求める
    def gradient(self, x, t):
        # TODO 未実装
