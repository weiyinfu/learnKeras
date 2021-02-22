"""
TimeDistributed的inputLayer的每个样本的形状至少是二维
"""
from keras.layers import *
from keras.models import *
import numpy as np
from keras import losses
from keras.metrics import *
m = Sequential()
"""
输入数据每个样本为5,2,输出变成5,3,建立了5个2*3的全连接
"""
m.add(TimeDistributed(Dense(3), input_shape=(5, 2)))
m.compile('adam', losses.categorical_crossentropy)
print(m.predict(np.ones((1, 5, 2))).shape)

m = Sequential()
"""
输入也可以是多维数据
表示有5*4个二维时间片,每个时间片都是长度为2的向量
"""
m.add(TimeDistributed(Dense(3), input_shape=(5, 4, 2)))
m.compile("adam", losses.categorical_crossentropy)
print(m.predict(np.ones((1, 5, 4, 2))).shape)

m = Sequential()
"""
如果输入样本是一张图片,那么可以对图片进行卷积
因为conv2D需要接受3个维度的图片作为输入
这里输入是5张(时序)4*5的深度为3的图片
"""
m.add(TimeDistributed(Conv2D(filters=1, kernel_size=(2, 2)), input_shape=(5, 4, 4, 7)))
m.compile("adam", losses.categorical_crossentropy)
print(m.predict(np.ones((1, 5, 4, 4, 7))).shape)
