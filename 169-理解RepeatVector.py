import keras

"""
在共享权值的时候，可以通过repeat或者RepeatVector来实现
"""
x = keras.backend.variable(keras.initializers.random_uniform()((1, 1)))
y = keras.layers.RepeatVector(3)(x)
yy = keras.backend.repeat(x, 3)
f = keras.backend.function([x], [y, yy])
print(f([[[5, ]]]))

up = keras.backend.update(x, [[4, ]])
print(keras.backend.get_value(up))
print(f([[[6, ]]]))
