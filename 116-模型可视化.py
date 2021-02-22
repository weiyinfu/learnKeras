from keras.utils import plot_model
from keras import *
from keras.layers import *

x = Input((4, 4, 1))

conv22 = Conv2D(3, (2, 2), padding='SAME', activation='relu', name="conv22")(x)
pool1 = MaxPool2D()(conv22)

conv33 = Conv2D(3, (3, 3), padding='SAME', activation='relu', name='conv33')(x)
pool2 = MaxPool2D()(conv33)

flat = concatenate([Flatten()(i) for i in (pool1, pool2)])
dense1 = Dense(10, activation='relu', name="dense1")(flat)
drop1 = Dropout(0.2, name="dropout1")(dense1)
dense2 = Dense(3, activation='softmax', name="dense2")(drop1)

m = models.Model(inputs=x, outputs=dense2)
plot_model(m, 'data/model.png')
