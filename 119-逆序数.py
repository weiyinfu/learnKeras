"""
给定一个数字序列keras能否学到逆序数

很难学到
"""
from keras import *
from keras.layers import *
from lib.callbacks import StopWhen
from lib.inverse_number import get_data


class StopWhen(keras.callbacks.Callback):
    """
    当精度达到一定程度的时候自动结束训练
    """

    def __init__(self, key='acc', value=1.0):
        self.value = value
        self.key = key

    def on_epoch_end(self, epoch, logs=None):
        print("epoch end", epoch, logs)
        if self.key in logs and logs.get(self.key) >= self.value:
            self.model.stop_training = True

class StopWhen(keras.callbacks.Callback):
    """
    当精度达到一定程度的时候自动结束训练
    """

    def __init__(self, key='acc', value=1.0):
        self.value = value
        self.key = key

    def on_epoch_end(self, epoch, logs=None):
        print("epoch end", epoch, logs)
        if self.key in logs and logs.get(self.key) >= self.value:
            self.model.stop_training = True

sz = 4
m = models.Sequential()
m.add(Dense(100, activation='sigmoid', input_shape=(sz,)))
m.add(Dense(100, activation='sigmoid', input_shape=(sz,)))
m.add(Dense(2, activation='softmax'))
m.compile(optimizers.Adam(lr=0.1), losses.categorical_crossentropy, ["acc"])
batch_size = 32
train_x, train_y, test_x, test_y = get_data(one_hot=True, sz=sz, shuffle=True)
print(train_x[:3], train_y[:3])
m.fit(train_x, train_y, batch_size=batch_size, epochs=int(1e9), callbacks=[StopWhen('acc', 1.0)], verbose=False)
score = m.evaluate(test_x, test_y)
print(score)
