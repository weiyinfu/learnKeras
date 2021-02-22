from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from lib.callbacks import StopWhen

class StopWhen(keras.callbacks.Callback):
    """
    当精度达到一定程度的时候自动结束训练
    """

    def __init__(self, key='acc', value=1.0):
        self.value = value
        self.key = key

    def on_epoch_end(self, epoch, logs=None):
        print("epoch end", epoch, logs)
        if self.key in logs and logs.get(self.key) >= self.value:
            self.model.stop_training = True

"""
乘幂全连接层
练习自定义层
"""


def wx(x, w):
    def wx_one(xi):
        # 可以改动的地方：wx的操作，和对reduce_sum还是reduce_mean还是reduce_prod
        return tf.map_fn(lambda wi: tf.reduce_sum(xi ** wi), tf.transpose(w, [1, 0]))

    return tf.map_fn(wx_one, x)


class PowerDense(keras.layers.Layer):
    def __init__(self, output_dim, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super(PowerDense, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='random_normal',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,),
                                    initializer='uniform',
                                    trainable=True)
        # self.built=True
        super(PowerDense, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        y = wx(x, self.kernel)
        y += self.bias
        if self.activation == 'sigmoid':
            y = keras.activations.sigmoid(y)
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

data = load_iris()
data.target = keras.utils.to_categorical(data.target)
train_x, test_x, train_y, test_y = train_test_split(data.data, data.target, train_size=0.8)

model = Sequential()
model.add(PowerDense(32, activation='sigmoid', input_shape=(train_x.shape[1],)))
model.add(PowerDense(train_y.shape[1]))
model.add(keras.layers.Softmax())
optimizer = keras.optimizers.Adam(lr=0.1)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'], )

model.fit(train_x, train_y,
          epochs=int(1e9),
          batch_size=len(train_x), callbacks=[StopWhen("acc",0.97)])
score = model.evaluate(test_x, test_y, batch_size=128)
print(score)
