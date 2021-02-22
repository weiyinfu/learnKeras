import matplotlib.pyplot as plt
from skimage.data import astronaut
import numpy as np
from keras.preprocessing import image

datagen = image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    # shear_range=0.3,
    zoom_range=0.3,
)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
x = np.array([astronaut()])
datagen.fit(x)

# here's a more "manual" example
fig, ax = plt.subplots(3, 3)
ax = ax.reshape(-1)
imgs = []
for x_batch, y_batch in datagen.flow(x, [[0]], batch_size=9):
    imgs.append(x_batch[0])
    if len(imgs) == len(ax): break
for img, axe in zip(imgs, ax):
    axe.imshow(img)
    axe.axis('off')
plt.show()
