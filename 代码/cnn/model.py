import numpy as np

from module import *


def randomly_load_data(X, y, batch_size):
    """从X,y中随机选取batch_num个组成列表返回"""
    num_samples = X.shape[0]
    indices = np.random.choice(num_samples, batch_size, replace=False)
    # 使用这些索引从X和y中抽取样本和标签
    batch_X = X[indices]
    batch_y = y[indices]
    return batch_X, batch_y


def softmax(z):
    # 首先，要调节一下z的值，避免z太大，使得exp报错
    p = np.zeros_like(z)
    batch_size = z.shape[0]
    for batch in range(batch_size):
        Z = z[batch] - np.max(z[batch])
        # 正式计算softmax
        exp_Z = np.exp(Z)
        p[batch] = exp_Z / np.sum(exp_Z)
    return p


def cross_entropy(p, y):
    ans = np.zeros(shape=(p.shape[0],))
    for i in range(p.shape[0]):
        ans[i] = -np.log(p[i][y[i]] + 0.000001)  # 加一个小量避免出零
    return ans


def cross_entropy_grad(p, y):
    P = p.copy()
    for i in range(P.shape[0]):
        P[i][y[i]] -= 1
    return P


class Model:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, x):
        """计算到z分数为止"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_Loss):
        """反向传播"""
        for layer in self.layers[::-1]:
            grad_Loss = layer.backward(grad_Loss)

    def train_single(self, X, y):

        Z = self.forward(X)
        P = softmax(Z)
        Loss = cross_entropy(P, y)
        Loss_grad = cross_entropy_grad(P, y)

        self.backward(Loss_grad)
        # 结束，把返回值输出回去吧
        return Loss.mean()

    def fit(self, X_list, y_list, epoch_num, batch_size, optimizer, print_Loss=False):
        LL = 0
        parameters = self.parameters()
        for epoch in range(epoch_num):

            batch_X, batch_y = randomly_load_data(X_list, y_list, batch_size)
            Loss = self.train_single(batch_X, batch_y)
            if print_Loss:
                print(f"epoch {epoch}/{epoch_num}, Loss={Loss}")

            optimizer.update(parameters)

            if np.max(np.abs(LL - Loss)) < 0.0000001:
                break
            LL = Loss

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

    def parameters(self):
        ans = []
        for layer in self.layers:
            ans.extend(layer.parameters())
        return ans
