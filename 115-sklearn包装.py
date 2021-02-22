from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import keras
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import *

data = load_iris()
data.target = keras.utils.to_categorical(data.target)
train_x, test_x, train_y, test_y = train_test_split(data.data, data.target, train_size=0.8)


def get_model():
    model = Sequential()
    # Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
    # 在第一层必须指定所期望的输入数据尺寸：
    # 在这里，是一个 20 维的向量。
    model.add(Dense(64, activation='relu', input_dim=train_x.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(train_y.shape[1], activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


clf = KerasClassifier(get_model)
clf.fit(train_x, train_y, epochs=200,
        batch_size=len(train_x))
score = clf.score(test_x, test_y)
print(score)
