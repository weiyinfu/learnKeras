import keras.backend as K
from keras import models
from keras.layers import *
from keras.optimizers import Adam
import numpy as np

"""
自定义层
"""


# add a layer that returns the concatenation
# of the positive part of the input and
# the opposite of the negative part

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)


def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)


model = models.Sequential()
model.add(Lambda(antirectifier,
                 output_shape=antirectifier_output_shape))
model.compile(optimizer=Adam())
res = model.predict(np.array([[1, 2, 3]]))
print(res)
