import tensorflow as tf
import keras.layers as l

x_place = tf.placeholder(dtype=tf.float32, shape=(None, 3))
y = l.Dense(3)(x_place)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    yy = sess.run(y, feed_dict={
        x_place: [[1, 2, 3]]
    })
    print(yy)
