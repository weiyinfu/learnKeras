import keras
from keras.preprocessing.sequence import pad_sequences, skipgrams, make_sampling_table
import matplotlib.pyplot as plt

"""
keras中的预处理函数都是纯粹的python函数
"""


def pad():
    # 填充序列
    s = pad_sequences([[1, 2], [3, 4, 5], [6, 7, 8, 9]], maxlen=7, )
    print(s)


def skip_grams():
    """
    生成sktip_gram训练集
    :return:
    """
    # 设置negative_smaples=0和shuffle=false能够使得结果更直观,只显示不排序的正例
    s = skipgrams([1, 2, 3, 4], vocabulary_size=10, window_size=3, seed=0, negative_samples=0, shuffle=False)
    print(s)
    x, y = s
    for xx, yy in zip(x, y):
        print(xx, yy)


def make_sample_table():
    # sampling_factor表示增长速率
    table = make_sampling_table(100, sampling_factor=0.01)
    print(table)
    plt.plot(table)
    plt.show()


# skip_grams()

make_sample_table()
