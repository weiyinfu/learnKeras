from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import Dense, Dropout
from keras.models import Sequential
from lib.inverse_number import get_data
from lib.callbacks import StopWhen

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

seq_length = 5
x_train, y_train, x_test, y_test = get_data(one_hot=False, shuffle=True, sz=seq_length)
x_train = x_train.reshape(-1, x_train.shape[1], 1)
x_test = x_test.reshape(-1, x_test.shape[1], 1)

model = Sequential()
model.add(Conv1D(64, 2, activation='relu', input_shape=(seq_length, 1), padding='SAME'))
model.add(Conv1D(64, 3, activation='relu', padding='SAME'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu', padding='SAME'))
model.add(Conv1D(128, 3, activation='relu', padding='SAME'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=100000, callbacks=[StopWhen('acc', 0.95)])
score = model.evaluate(x_test, y_test, batch_size=16)
print(score)