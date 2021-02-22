import tensorflow as t

import keras.backend as k

"""
有gradients求导函数，有tf.assign赋值函数，我们就可以自己实现Optimizer了
"""
x = t.Variable(4, dtype=t.float32)
y = x * x + 3 * x + 4  # 导数为2x+3
g1 = t.gradients(y, x)
g2 = k.gradients(y, x)
print(g1, g2)
with k.get_session() as sess:
    m, n = sess.run([g1, g2])
    print(m)
    print(n)
