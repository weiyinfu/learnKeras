from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt

data = load_iris()
data.target = keras.utils.to_categorical(data.target)
train_x, test_x, train_y, test_y = train_test_split(data.data, data.target, train_size=0.8)
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(train_y.shape[1], activation='sigmoid'))
"""
optimizer也可以用字符串指定
"""
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
"""
训练轮数虽然多,但是要求准确率达到1的时候退出程序
"""


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("train_begin", logs)

    def on_train_end(self, logs=None):
        print("train_end", logs)

    def on_epoch_begin(self, epoch, logs=None):
        print("epoch begin", epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        print("epoch end", epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        print("batch begin", batch, logs)

    def on_batch_end(self, batch, logs=None):
        print("betch end", batch, logs)
        if logs.get('acc') > 0.99:
            self.model.stop_training = True


print(dir(model))

history = model.fit(train_x, train_y,
                    epochs=2000,
                    batch_size=min(train_x.shape[0], 128),
                    callbacks=[LossHistory()], verbose=0)
score = model.evaluate(test_x, test_y, batch_size=min(test_x.shape[0], 128))
print(score)
plt.plot(history.history.get('acc'))
plt.plot(history.history.get('loss'))
plt.show()
