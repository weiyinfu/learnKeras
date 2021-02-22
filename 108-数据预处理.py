import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import keras

"""
正则化是对每个样本进行正则化
每个样本相遇向量，对每个向量进行归一化
"""
a = np.array([[1, 2], [2, 1.0], [3, 4]])  # 10个样本
print('手写正则化\n', a / np.tile(np.linalg.norm(a, 2, axis=1).reshape(-1, 1), (1, a.shape[1])))
print("keras 正则化\n", keras.utils.normalize(a.copy(), axis=1))
print("normalize\n", Normalizer().fit_transform(a.copy()))

"""
标准化：减去均值、除以标准差
"""
print('StandardScaler\n', StandardScaler().fit_transform(a.copy()))
print("手写缩放\n", (a - np.mean(a, axis=0)) / np.std(a, axis=0))

"""
尺度缩放：默认缩放到0到1之间
"""
print("minmaxScaler", MinMaxScaler().fit_transform(a.copy()))
print("手写minmaxScaler", (a - np.min(a, axis=0)) / (np.max(a, axis=0) - np.min(a, axis=0)))

"""
理解np.std
标准差
"""

print('np.std', np.std(a, axis=0))
print('手写std', np.mean((a - np.mean(a, axis=0)) ** 2, axis=0) ** 0.5)
