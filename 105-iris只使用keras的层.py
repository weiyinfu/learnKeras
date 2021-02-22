import numpy as np
import tensorflow as T
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from keras import *
from keras import backend as tf

data = load_iris()
data.target = utils.to_categorical(data.target)
print(data.target.shape)
train_x, test_x, train_y, test_y = train_test_split(data.data, data.target)

x_place = tf.placeholder(dtype=np.float32, shape=(None, train_x.shape[1]))
y_place = tf.placeholder(dtype=np.float32, shape=(None, train_y.shape[1]))

dense1 = layers.Dense(10, activation='relu')(x_place)
# 中间加上一层tensorflow的全连接
w = T.Variable(T.random_normal((10, 20)))
b = T.Variable(T.random_normal((20,)))
dense1 = T.matmul(dense1, w) + b
dense1 = layers.Activation("relu")(dense1)
logits = layers.Dense(3)(dense1)

loss = tf.sum(tf.categorical_crossentropy(y_place, logits, True))
acc = tf.mean(tf.cast(tf.equal(tf.argmax(y_place, axis=1), tf.argmax(logits, axis=1)), np.float32))
train = T.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
with tf.get_session() as sess:
    sess.run(T.global_variables_initializer())
    for i in range(100):
        _, l, a = sess.run([train, loss, acc], feed_dict={
            x_place: train_x,
            y_place: train_y
        })
        print(i, l, a)
