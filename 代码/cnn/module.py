import numpy as np
from Optimizer import Parameter


class Layer:
    """虚基类。一个层，可能是线性层、卷积层、池化层什么的"""

    def __init__(self):
        """层基类初始化方法"""
        pass

    def forward(self, x):
        """层基类前向传播"""
        pass

    def backward(self, *args):
        """层基类反向传播"""
        pass

    def parameters(self):
        """返回层的所有参数组成的列表"""
        return []


class LinearLayer(Layer):
    """线性层，输入输出必须为1维数组。不包含偏置层。"""

    def __init__(self, input_size: int, output_size: int):
        """
        初始化参数:
        input_size:输入向量的维度数，必须为int
        output_size:输出向量的维度数，必须为int

        输入形状：[batch_size,input_size]
        输出矩阵：[batch_size,output_size]
        """
        super(LinearLayer, self).__init__()  # 调用基类初始化
        self.input_size = input_size
        self.output_size = output_size
        self.last_input = None
        self.weight = Parameter(shape=(output_size, input_size))  # 线性层的权重

    def forward(self, x):
        """线性层前向传播"""
        # x:[batch_size,input_size]
        self.last_input = x
        ans = x.dot(self.weight.value.transpose())
        return ans

    def backward(self, grad_Loss):
        """线性层反向传播"""
        # grad_Loss:[batch_size,output_size]
        # x:[batch_size,input_size]
        # weight:[o,i]
        self.weight.grad += grad_Loss.transpose().dot(self.last_input)
        x_grad = grad_Loss.dot(self.weight.value)
        return x_grad

    def parameters(self):
        """线性层的参数列表只有权重矩阵"""
        return [self.weight]


class BiasLayer(Layer):
    """偏置层，输入输出形状任意但相等"""

    def __init__(self, shape: tuple):
        """
        初始化参数:
        shape: 输入输出数组的shape

        输入输出形状任意但相等
        """
        super(BiasLayer, self).__init__()  # 调用基类初始化
        self.shape = shape
        self.bias = Parameter(shape)

    def forward(self, x):
        """偏置层前向传播"""
        # 由于广播机制的存在，[batch_size,shape]可以直接与[shape]正确地相加
        return x + self.bias.value

    def backward(self, grad_Loss):
        """偏置层反向传播"""
        self.bias.grad += grad_Loss.sum(axis=0)  # G
        return grad_Loss

    def parameters(self):
        """偏置层的参数列表只有偏置数组"""
        return [self.bias]


class ConvolveLayer(Layer):
    """卷积层，不进行padding，输出尺寸会比输入尺寸小。输入输出必须是正方形"""

    def __init__(self,
                 input_size,
                 in_channels,  # 输入通道数
                 out_channels,  # 输出通道个数
                 kernel_size,  # 卷积核边长
                 ):
        """
        初始化参数:
        input_size: 输入矩阵的边长
        in_channels: 输入通道数
        out_channels: 输出通道个数
        kernel_size: 卷积核边长

        输出矩阵边长=输入矩阵边长-卷积核边长+1

        输入形状：[batch_size,in_channels,行,列]
        输出形状：[batch_size,out_channels,行,列]
        """
        super(ConvolveLayer, self).__init__()  # 调用基类初始化
        self.kernels = Parameter(shape=(out_channels, in_channels, kernel_size, kernel_size))
        # 其中self.kernels[o,i]是第i个输入给第j个输出的贡献
        self.kernel_size = kernel_size
        self.last_input = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_size = input_size - self.kernel_size + 1

    def forward_old(self, x):
        """这是一个较老版本的forward函数，它可以很好地展示原理，但是计算慢"""
        # 输入格式：[batch_size,通道选择,行,列]
        # 输出格式：[batch_size,通道选择,行,列]

        self.last_input = x
        batch_size = x.shape[0]
        y = np.zeros(shape=(batch_size, self.out_channels, self.out_size, self.out_size))
        k = self.kernel_size
        for i in range(self.out_size):
            for j in range(self.out_size):
                for batch in range(batch_size):
                    for o_channel in range(self.out_channels):
                        for i_channel in range(self.in_channels):
                            x_piece = x[batch, i_channel, i:i + k, j:j + k]
                            kernel = self.kernels.value[o_channel, i_channel]
                            y[batch, o_channel, i, j] += (x_piece * kernel).sum()

        return y

    def forward(self, x):
        """卷积层前向传播，执行卷积操作"""
        # 输入格式：[batch_size,通道选择,行,列]
        # 输出格式：[batch_size,通道选择,行,列]

        self.last_input = x
        bath_size = x.shape[0]
        y = np.zeros(shape=(bath_size, self.out_channels, self.out_size, self.out_size))

        for i in range(self.out_size):
            for j in range(self.out_size):
                # for all batch,in_channel:
                y[:, :, i, j] += np.einsum(
                    "bimn,oimn->bo",
                    x[:, :, i:i + self.kernel_size, j:j + self.kernel_size],  # x[batch,i_channel,K,K]
                    self.kernels.value  # [o_channel,i_channel,K,K]
                )

        return y

    def backward_old(self, grad_Loss):
        """这是一个较老版本的反向传播函数，它可以很好地展示原理，但是计算慢"""

        batch_size = grad_Loss.shape[0]

        x = self.last_input
        # 向x反向传播
        # v1
        k = self.kernel_size
        K = self.kernels.value

        x_grad2 = np.zeros_like(self.last_input)
        K_grad2 = np.zeros_like(self.kernels.grad)

        for i in range(self.out_size):
            for j in range(self.out_size):
                for batch in range(batch_size):
                    for o_channel in range(self.out_channels):
                        for i_channel in range(self.in_channels):
                            x_grad2[batch, i_channel, i:i + k, j:j + k] += grad_Loss[batch, o_channel, i, j] * K[
                                o_channel, i_channel]

        for i in range(self.out_size):
            for j in range(self.out_size):
                for batch in range(batch_size):
                    for o_channel in range(self.out_channels):
                        for i_channel in range(self.in_channels):
                            K_grad2[o_channel, i_channel] += \
                                grad_Loss[batch, o_channel, i, j] * x[batch, i_channel, i:i + k, j:j + k]
        self.kernels.grad = K_grad2

        return x_grad2

    def backward(self, grad_Loss, experimental=False):
        """卷积层反向传播，计算Loss关于卷积核和x的梯度。einsum大大地提高了计算效率"""

        x = self.last_input
        # 向x反向传播
        k = self.kernel_size
        K = self.kernels.value

        x_grad = np.zeros_like(self.last_input)

        for i in range(self.out_size):
            for j in range(self.out_size):
                grad_value = grad_Loss[:, :, i, j]
                x_grad[:, :, i:i + k, j:j + k] += np.einsum("bo,oimn->bimn", grad_value, K)
                self.kernels.grad += np.einsum("bimn,bo->oimn", x[:, :, i:i + k, j:j + k], grad_value)

        return x_grad

    def parameters(self):
        """卷积层的参数列表，即卷积核"""
        return [self.kernels]


class SigmoidLayer(Layer):
    """Sigmoid激活函数"""
    def __init__(self):
        """无初始化参数，输入输出形状任意但需相等"""
        super(SigmoidLayer, self).__init__()  # 调用基类初始化
        self.last_output = None

    def forward(self, x):
        """sigmoid函数体"""
        self.last_output = 1.0 / (1 + np.exp(-x))
        return self.last_output

    def backward(self, grad_Loss):
        """反向传播，计算Loss关于x的梯度"""
        # s'(x) = s(x)(1-s(x))
        return self.last_output * (1-self.last_output) * grad_Loss

    # Sigmoid层没有参数，直接继承基类的parameters函数


class ReshapeLayer(Layer):
    """这个层负责将输入的矩阵变形成所需形状的输出矩阵，从而实现卷积层和线性层的对接"""
    def __init__(self, From, To):
        """
        初始化参数:
        From: 输入矩阵的形状
        To: 输出矩阵的形状
        """
        super(ReshapeLayer, self).__init__()  # 调用基类初始化
        self.From = (-1,) + From
        self.To = (-1,) + To

    def forward(self, x):
        """将输入矩阵变形为输出矩阵的形状"""
        return x.reshape(self.To)

    def backward(self, grad_Loss):
        """将反向传播梯度矩阵变形为输入矩阵的形状"""
        return grad_Loss.reshape(self.From)

    # Reshape层没有参数，直接继承基类的parameters函数


class PoolingLayer(Layer):
    """池化层，步长和窗口边长相等"""
    def __init__(self, pooling_size):
        """
        初始化参数：pooling_size：池化层窗口边长
        输入形状：[batch_size,channel_num,行,列]
        输出形状：[batch_size,channel_num,行,列]
        """
        super(PoolingLayer, self).__init__()  # 调用基类初始化
        self.x = None
        self.pool_size = pooling_size

    def forward(self, x):
        """对多通道二维图像进行平均池化，输出通道与输入通道保持一致"""
        self.x = x
        batch_size, channel_num, H, W = x.shape
        n = self.pool_size

        pooled_image = np.zeros((batch_size, channel_num, H // n, W // n), dtype=x.dtype)
        for i in range(0, H // n):
            for j in range(0, W // n):
                pooled_image[:, :, i, j] = np.mean(
                    x[:, :, i * n:(i + 1) * n, j * n:(j + 1) * n], axis=(2, 3))

        return pooled_image

    def backward(self, grad_Loss):
        """池化层反向传播，将下一次传过来的梯度平均分给参与构成该单元格的各个输入"""
        x_grad = np.zeros_like(self.x)
        batch_size, channel_num, H, W = self.x.shape
        n = self.pool_size
        for i in range(0, H // n):
            for j in range(0, W // n):
                x_grad[:, :, i * n:(i + 1) * n, j * n:(j + 1) * n] = (grad_Loss[:, :, i:i + 1, j:j + 1] / (n ** 2))
        return x_grad
    # Pooling层没有参数，直接继承基类的parameters函数
