import pandas as pd

from model import *
from module import *
from optim import *
import matplotlib.pyplot as plt

# 设置随机种子，保证运行结果的确定性
np.random.seed(0)
# 从csv文件中读文件
data_frame = pd.read_csv(".\\train.csv")
data = data_frame.to_numpy()

# 切分训练测试集
num_train = int(0.9 * data.shape[0])
permuted_indices = np.random.permutation(data.shape[0])  # 随机重排
train_indices = permuted_indices[:num_train]
test_indices = permuted_indices[num_train:]
train_data = data[train_indices]
test_data = data[test_indices]

# 将data切割为X,y
train_X = train_data[:, 1:].reshape((-1, 1, 28, 28)) / 256  # 除256归一化

train_y = train_data[:, 0]
test_X = test_data[:, 1:].reshape((-1, 1, 28, 28)) / 256  # 除256归一化
test_y = test_data[:, 0]



model = Model(layers=[
    ConvolveLayer(input_size=28, in_channels=1, out_channels=6, kernel_size=5),
    BiasLayer(shape=(6, 24, 24)),
    SigmoidLayer(),
    PoolingLayer(pooling_size=2),

    ConvolveLayer(input_size=12, in_channels=6, out_channels=16, kernel_size=5),
    BiasLayer(shape=(16, 8, 8)),
    SigmoidLayer(),
    PoolingLayer(pooling_size=2),
    ReshapeLayer(From=(16, 4, 4), To=(256,)),
    LinearLayer(input_size=256, output_size=80),
    SigmoidLayer(),
    LinearLayer(input_size=80, output_size=10),

])

model.fit(
    X_list=train_X,
    y_list=train_y,
    epoch_num=1000,
    batch_size=64,
    print_Loss=True,
    optimizer=Adam(learning_rate=0.01)
)



valid_frame = pd.read_csv(".\\test.csv")
valid_data = valid_frame.to_numpy()
valid_X = valid_data.reshape((-1, 1, 28, 28)) / 256

print("testing...")
# predict = model.predict(valid_X)
predict = model.predict(test_X)
print(predict)
print(test_y)

submission = pd.DataFrame({
    'ImageId': range(1, len(predict) + 1),
    'Label': predict
})
submission.to_csv('submission2.csv', index=False)

equal = predict == test_X
assert isinstance(equal, np.ndarray)
accuracy = equal.sum() / equal.size

print(accuracy)
