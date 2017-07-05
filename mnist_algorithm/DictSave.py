# -*- coding:utf-8 -*-
import pickle,pprint
from PIL import Image
import numpy as np
import os
import matplotlib.image as plimg
class DictSave(object):
    def __init__(self,filenames):
        self.filenames = filenames
        self.arr = []
        self.all_arr = []
        print
    def image_input(self,filenames):
        for filename in  filenames:
            self.arr = self.read_file(filename)
            if self.all_arr==[]:
                self.all_arr = self.arr
            else:
                self.all_arr = np.concatenate((self.all_arr,self.arr))
    def read_file(self,filename):
        im = Image.open(filename)#打开一个图像
        # 将图像的RGB分离
        r, g, b = im.split()
        # 将PILLOW图像转成数组
        r_arr = plimg.pil_to_array(r)
        g_arr = plimg.pil_to_array(g)
        b_arr = plimg.pil_to_array(b)
        # 将32*32二位数组转成1024的一维数组
        r_arr1 = r_arr.reshape(1024)
        g_arr1 = g_arr.reshape(1024)
        b_arr1 = b_arr.reshape(1024)
        # 3个一维数组合并成一个一维数组,大小为3072
        arr = np.concatenate((r_arr1, g_arr1, b_arr1))
        return arr
    def pickle_save(self,arr):
        print "正在存储"
        # 构造字典,所有的图像诗句都在arr数组里,我这里是个以为数组,目前并没有存label
        contact = {'data': arr}
        f = open('contact', 'w')
        pickle.dump(contact, f)#把字典存到文本中去
        f.close()
        print "存储完毕"
if __name__ == "__main__":
    rootdir = 'createdImg/'
    filenames = os.listdir(rootdir)

    ds = DictSave(filenames)
    ds.image_input(ds.filenames)
    ds.pickle_save(ds.all_arr)
    print "最终数组的大小:"+str(ds.all_arr.shape)