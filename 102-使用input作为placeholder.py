from keras import *
import tensorflow as tf

x = layers.Input((1,))
y = 2 * x
with tf.Session() as sess:
    print(sess.run(y, feed_dict={x: [[3]]}))
