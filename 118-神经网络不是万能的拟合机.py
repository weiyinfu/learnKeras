import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import keras
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential

"""
随机给定单位圆上一个点，求这个点对应的角度
当这个点无限接近(1,0)时，角度在360和0之间突变，这就导致神经网络在这紧要关头学习不好。
谁有能说MNIST、IRIS数据集所对应的分布不包含突变呢？
一旦找到了突变的“那个点”，我们能够做什么？
能够对神经网络进行攻击，想让它变成什么值就变成什么值。
在这个问题中，(1,0)点附近的突变使得我们伪造数据，让神经网络输出任意结果。

这就像：对抗学习中总能够找到一个不像“狗”的狗的图片
"""
np.random.seed(42)
tf.set_random_seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def get_xy(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


# 训练集
point_n = 1000
points = np.zeros([point_n, 2])
point_theta = np.zeros([point_n, ])
for i in range(point_n):
    theta = np.random.rand() * 2 * np.pi
    x, y = get_xy(1, theta)
    points[i, :] = x, y
    point_theta[i] = theta
# 模型
model = Sequential()
model.add(Dense(4, input_shape=(2,)))
model.add(Dense(8, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(2, activation="relu"))
model.add(Dense(1))
model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam())
model.summary()
model.fit(points, point_theta, batch_size=32, verbose=1, epochs=200, shuffle=True)

# 预测角度
test_points = np.zeros([3600, 2])
for i in range(3600):
    test_points[i, :] = get_xy(1, 0.1 * i / 180 * np.pi)
x_pred = model.predict(test_points)
plt.xlim([-10, 360 + 10])
plt.plot(range(360), range(360), c="b")
plt.plot(np.array(range(3600)) * 0.1, x_pred * 180 / np.pi, c="r")
plt.savefig("data/拓扑变换360.png")
plt.clf()# 清空一下，准备下次画图

# 预测角度 0度附近
test_points = np.zeros([100, 2])
for i in range(100):
    test_points[i, :] = get_xy(1, (-1 + 0.02 * i) / 180 * np.pi)
x_pred = model.predict(test_points)
plt.xlim([-1, 1])
plt.plot(np.array(range(100)) * 0.02 - 1, x_pred * 180 / np.pi, c="r")
plt.savefig("data/拓扑变换0.png")
