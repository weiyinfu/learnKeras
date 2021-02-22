from itertools import permutations

import numpy as np
from  sklearn.model_selection import train_test_split

import keras
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping

"""
逆序数生成器，x为一个随机排列，y为这个排列逆序数的奇偶性
"""


def get_ans(x):
    return np.array([sum(np.count_nonzero(x[i][j:] > x[i][j]) for j in range(len(x[i]))) & 1 for i in range(len(x))])


def get_data(one_hot=False, sz=5, shuffle=False):
    x = np.array(list(permutations(list(range(sz)))))
    y = get_ans(x)
    train_x, test_x, train_y, test_y = train_test_split(x, y, shuffle=shuffle)

    if one_hot:
        train_yy = keras.utils.to_categorical(train_y)
        test_yy = keras.utils.to_categorical(test_y)
    else:
        train_yy = train_y
        test_yy = test_y
    return train_x, train_yy, test_x, test_yy


seq_length = 5
x_train, y_train, x_test, y_test = get_data(one_hot=False, shuffle=True, sz=seq_length)

"""
如果包含偶数个1,输出0
包含奇数个1,输出1
"""
max_features = seq_length
model = Sequential()
model.add(Embedding(input_dim=1, output_dim=seq_length))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=int(1e9), callbacks=[EarlyStopping(monitor='acc', min_delta=0.01, patience=3, mode='max')])
score = model.evaluate(x_test, y_test, batch_size=16)
print(score)
