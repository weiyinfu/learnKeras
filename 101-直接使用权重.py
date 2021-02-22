from keras import *

"""
layer.get_weights()直接获得numpy数组
"""
l = layers.Dense(3)
print(l.get_weights())  # 没有指定输入,所以是个3*0的矩阵

l(layers.Input((2,)))  # 3*2的矩阵
print(l.get_weights())

"""
layers反序列化有两种方式:
* ll = layers.Dense.from_config(l.get_config())
* ll = layers.deserialize({'class_name': layer.__class__.__name__,'config': config})
layers.config只保存结构信息,不保存权重信息
"""
print(l.get_config())
ll = layers.Dense.from_config(l.get_config())
ll(layers.Input((2,)))
print(ll.get_weights())


ll = layers.deserialize(
    dict(class_name=l.__class__.__name__,
         config=l.get_config())
)
print(ll.get_config())