"""
keras自带了一些列数据集，本程序对这些数据集进行一些可视化展示
"""

from keras import datasets
import matplotlib.pyplot as plt
import math

(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data("/data/mnist.npz")


def show_many(imgs):
    rows = int(len(imgs) ** 0.5)
    cols = math.ceil((len(imgs) / rows))
    fig, ax = plt.subplots(rows, cols)
    ax = ax.reshape(-1)
    for im, axe in zip(imgs, ax):
        axe.imshow(im)
    for axe in ax:
        axe.axis('off')
    plt.show()


show_many(train_x[:10])

(train_x, train_y), (test_x, test_y) = datasets.fashion_mnist.load_data("/data/fashion_mnist")
show_many(train_x[:10])
