import numpy as np

global_key = 0


class Parameter:
    """一个ndarray参数，同时绑定一个梯度值，将两者绑定在一起管理"""

    def __init__(self, shape):
        """
        初始化参数：shape: 参数的形状
        参数值将会被随机初始化，服从正态分布(均值=0,方差=1)
        梯度将会被初始化为全0

        参数对象将会被分配一个唯一的key，用于被优化器识别身份
        """
        global global_key
        self.shape = shape
        self.value = np.random.normal(size=shape)  # 值
        self.grad = np.zeros_like(self.value)  # 关于Loss的梯度

        self.key = global_key
        global_key += 1

    def zero_grad(self):
        """将梯度清零"""
        self.grad[:] = 0.0


class Adam:
    """Adam优化器"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.99, epsilon=1e-8):
        super(Adam, self).__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 第一阶矩估计（相当于速度）
        self.v = {}  # 第二阶矩估计（相当于加速度）
        self.t = 0  # 时间步

    def update(self, params: list[Parameter]):
        self.t += 1

        for para in params:
            key = para.key
            if key not in self.m:
                self.m[key] = np.zeros_like(para)
                self.v[key] = np.zeros_like(para)

                # 更新一阶矩估计（动量）
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * para.grad
            # 更新二阶矩估计（速度的平方）
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(para.grad)

            # 偏差修正
            m_hat = self.m[key] / (1 - np.power(self.beta1, self.t))
            v_hat = self.v[key] / (1 - np.power(self.beta2, self.t))

            # 更新参数
            para.value -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            para.zero_grad()
