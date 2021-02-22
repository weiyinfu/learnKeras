import keras.backend as K
import keras
import tensorflow as tf

"""
keras的function可以方便的求某几个数字的值
"""
input = keras.layers.Input((None,))
output = tf.multiply(input, input)
output2 = keras.layers.multiply([input, input])
called_count = K.variable(0.0)
f = K.function([input], [output, output2, called_count], [K.update_add(called_count, 1)])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(f([[3, 4, 5]]))
    print(f([[3, 4, 5]]))
    o, oo, c = sess.run([output, output2, called_count], feed_dict={
        input: [[3, 4, 5]]
    })
    print(o, oo, c)
