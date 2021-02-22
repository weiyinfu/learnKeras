import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model

"""
VGG是一种序贯模型的神经网络
卷卷池弃
卷卷池弃
平全弃全

VGG的特点是:只使用3*3的卷积核

本程序示意一下VGG的小型版本


以19层VGG模型为例

两次使用conv3-64

两次使用conv3-128

四次使用conv3-256

八次使用conv3-512

经过卷积后，图片尺寸为7*7

所以参数总数为(3*3*64+64)*2+(3*3*128+128)*2+(3*3*256+256)*4+(3*3*512+512)*8+...

    (512*7*7*4096+4096)+(4096*4096+4096)+(4096*1000+1000)=  123697896

每个参数占四个字节

模型大小为   494791584Byte=471.87M

其中第一个全连接层占所有模型大小的 (512*7*7*4096+4096)/123697896=0.8308

"""
# 生成虚拟数据100张长宽均为100*100的RGB图像
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# 输入: 3 通道 100x100 像素图像 -> (100, 100, 3) 张量。
# 使用 32 个大小为 3x3 的卷积滤波器。
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
plot_model(model, 'data/vgg.png')
model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
