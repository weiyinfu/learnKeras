import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import keras.activations as act

"""
keras中的激活函数其实就是用tensorflow实现的一堆函数
"""
x_data = np.linspace(-20, 20, 100).reshape(1, -1)
x_place = tf.constant(x_data)
activations = [act.softmax, act.sigmoid, act.relu, act.tanh, act.linear, act.elu, act.hard_sigmoid, act.softplus, act.softsign]
y = [tf.reshape(f(x_place), (-1,)) for f in activations]

with tf.Session() as sess:
    yy = sess.run(y)
    rows = int(np.sqrt(len(activations)))
    cols = int(np.ceil(len(activations) / rows))
    fig, axes = plt.subplots(rows, cols)
    for y_data, ax, f in zip(yy, axes.reshape(-1), activations):
        ax.plot(x_data.reshape(-1), y_data)
        ax.set_title(f.__name__)
    plt.show()
