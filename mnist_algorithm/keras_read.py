# coding=utf-8
import cv2
import os
import numpy as np

#导入各种用到的模块组件
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
import numpy as np
from PIL import Image
from keras import backend as k


#统计样本的数目
def __getnum__(path):
    fm=os.listdir(path)
    i=0
    for f in fm:
        i+=1
    return i      

#生成X,Y列
def __data_label__(path,count): 
    MapDict={"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"a":10,"b":11,"c":12,"d":13,"e":14,"f":15,"g":16,"h":17,
    "i":18,"j":19,"k":20,"l":21,"m":22,"n":23,"o":24,"p":25,"q":26,"r":27,"s":28,"t":29,"u":30,"v":31,"w":32,"x":33,"y":34,"z":35,
    "A":36,"B":37,"C":38,"D":39,"E":40,"F":41,"G":42,"H":43,"I":44,"J":45,"K":46,"L":47,"M":48,"N":49,"O":50,"P":51,"Q":52,"R":53,
    "S":54,"T":55,"U":56,"V":57,"W":58,"X":59,"Y":60,"Z":61}
    
    
    data = np.empty((count,1,28,28),dtype="float32")
    label = np.empty((count,),dtype="uint8")
    i=0;
    filename= os.listdir(path)
    for ff in filename :
        #print ff
        img = cv2.imread(path+"/"+ff,0)
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        
        fy = ff.split("_")[0]
        label[i]= MapDict[fy]
        i+=1
    return data,label

###############
#开始建立CNN模型
###############

#生成一个model
def __CNN__(testdata,testlabel,traindata,trainlabel):
    model = Sequential()

    #第一个卷积层，4个卷积核，每个卷积核大小5*5。1表示输入的图片的通道,灰度图为1通道。
    model.add(Convolution2D(20 , 5 , 5, border_mode='valid',input_shape=(1,28,28))) 
    model.add(Activation('sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #第二个卷积层，30个卷积核，每个卷积核大小5*5。
    #采用maxpooling，poolsize为(2,2)
    model.add(Convolution2D(30 , 5 , 5, border_mode='valid'))
    model.add(Activation('sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #第三个卷积层，16个卷积核，每个卷积核大小3*3
    #激活函数用tanh
    #采用maxpooling，poolsize为(2,2)
    #model.add(Convolution2D(16 , 3 , 3, border_mode='valid')) 
    #model.add(Activation('tanh'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, init='normal'))
    model.add(Activation('sigmoid'))

    #Softmax分类，输出是4类别
    model.add(Dense(62, init='normal'))
    model.add(Activation('softmax'))

    #############
    #开始训练模型
    ##############
    #使用SGD + momentum冲量
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile里的参数loss就是损失函数(目标函数)  
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
    #开始训练， show_accuracy在每次迭代后显示正确率 。  batch_size是每次带入训练的样本数目 ， nb_epoch 是迭代次数，  shuffle 是打乱样本随机。  
    model.fit(traindata, trainlabel, batch_size=16,nb_epoch=5,shuffle=True,verbose=1,show_accuracy=True,validation_data=(testdata, testlabel))
    #设置测试评估参数，用测试集样本
    model.evaluate(testdata, testlabel, batch_size=16,verbose=1,show_accuracy=True)


if __name__ == '__main__':    
    trainpath = 'outImgs/'
    testpath = 'test/'
    testcount=__getnum__(testpath)
    traincount=__getnum__(trainpath)
    testdata,testlabel= __data_label__(testpath, testcount)
    #print testlabel
    traindata,trainlabel= __data_label__(trainpath, traincount)
    
    #label为0~3共4个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
    testlabel = np_utils.to_categorical(testlabel, 62)
    trainlabel = np_utils.to_categorical(trainlabel, 62)
    
    __CNN__(testdata, testlabel, traindata, trainlabel)
    
    print "Done"