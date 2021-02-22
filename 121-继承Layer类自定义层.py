from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import keras


class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


a = keras.Sequential()
a.add(MyLayer(3))
a.compile(keras.optimizers.Adam())
print(a.predict(np.array([[1, 2, 3]])))
print(np.dot([[1, 2, 3]], a.get_layer(index=0).get_weights()))
