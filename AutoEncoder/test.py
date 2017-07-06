#coding=utf-8
'''
Created on 2016年12月3日
@author: chunsoft
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 导入 MNIST 数据
import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)


print mnist.train.images.shape  #(55000, 784)
print mnist.test.images.shape   #(10000, 784)


print mnist.train.images[0]

