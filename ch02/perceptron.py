import numpy as np

# x1,x2:入力 w1,w2:重み b:バイアス
def perceptron(x1,x2,w1,w2,b):
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    tmp = np.sum(w*x) + b
    return 0 if tmp <= 0 else 1

def AND(x1,x2):
    return perceptron(x1,x2,0.5,0.5,-0.7)

def NAND(x1,x2):
    return perceptron(x1,x2,-0.5,-0.5,0.7)

def OR(x1,x2):
    return perceptron(x1,x2,0.5,0.5,0.0)

def XOR(x1,x2):
    # 多層パーセプトロン
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

print("AND")
print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))

print("NAND")
print(NAND(0,0))
print(NAND(0,1))
print(NAND(1,0))
print(NAND(1,1))

print("OR")
print(OR(0,0))
print(OR(0,1))
print(OR(1,0))
print(OR(1,1))

print("XOR")
print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))