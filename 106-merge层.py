"""
layers里面的一些接受多个输入的工具
这是为了函数式模型的使用方便
"""
import keras
import tensorflow as tf

x = tf.constant([1, 2, 13], dtype=tf.float32)
y = tf.constant([4, 5, 6], dtype=tf.float32)
with tf.Session() as sess:
    print(sess.run([keras.layers.Add()([x, y])]))
    print(sess.run([keras.layers.Multiply()([x, y])]))
    print(sess.run([keras.layers.Subtract()([x, y])]))
    print(sess.run([keras.layers.Average()([x, y])]))
    print(sess.run([keras.layers.Maximum()([x, y])]))
    print(sess.run([keras.layers.Minimum()([x, y])]))
    print(sess.run([keras.layers.Concatenate()([x, y])]))

    # 除了使用类,也可以直接使用函数
    print(sess.run([keras.layers.add([x, y])]))
