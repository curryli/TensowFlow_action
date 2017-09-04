# -*- coding: utf-8 -*-
#VGG在mnist上跑是跑通了，不过基本没效果，因为mnist图片太小，不适合用VGG， VGG 跑imagenet很有效
# load MNIST data
import input_data
import numpy as np
mnist = input_data.read_data_sets("data/", one_hot=True)

# start tensorflow interactiveSession
import tensorflow as tf

import os                          #python miscellaneous OS system tool 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

 
BS = 16  #对于VGG来说，BS不能太大，否则gpu内存吃不消
runEpoch=50

X_train = mnist.train._images
y_train = mnist.train._labels



train_size = int(X_train.shape[0])

learning_rate=0.001

img_h = 28
img_w = 28
img_d = 1
 
n_classes=10
kb=0.8

def get_next_batch(batch_size, index_in_epoch, contents, labels):
    start = index_in_epoch
    index_in_epoch = index_in_epoch + batch_size
    end = index_in_epoch
    if index_in_epoch<train_size:
        return contents[start:end], labels[start:end]
    else:
        return contents[train_size-BS:train_size], labels[train_size-BS:train_size]

 



def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation

def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)   #集成功能， 与 tf.nn.relu(tf.add(tf.matmul(dense1, wd1),bd1))类似，不过这个仅限于全连接层这种一维的层，因为输入必须是2D的
        p += [kernel, biases]
        return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


def inference_op(input_op, keep_prob):  #对应PDF的109页的D列模型
    p = []
    # assume input_op shape is 28x28x1

    # block 1 -- outputs 112x112x64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2)

    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2)

    # # block 3 -- outputs 28x28x256
    conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2,  name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)    
    pool3 = mpool_op(conv3_3,   name="pool3",   kh=2, kw=2, dh=2, dw=2)

    # block 4 -- outputs 14x14x512
    conv4_1 = conv_op(pool3,    name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2,  name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3,   name="pool4",   kh=2, kw=2, dh=2, dw=2)

    # block 5 -- outputs 7x7x512
    conv5_1 = conv_op(pool4,    name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2,  name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3,   name="pool5",   kh=2, kw=2, dw=2, dh=2)

    # flatten
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    fc8 = fc_op(fc7_drop, name="fc8", n_out=n_classes, p=p)   #最后一层是不是不应该有relu？
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p


x = tf.placeholder(tf.float32,[None,img_h,img_w,img_d])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob=tf.placeholder(tf.float32)

y_pred, y_softmax, fc8, p = inference_op(x, kb)

cross_entropy=tf.reduce_mean(tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_softmax), reduction_indices=[1])))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct=tf.equal(tf.argmax(y_softmax,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct,"float"))

 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for e in range(runEpoch):
        ##每一轮重新打乱数据 
        perm = np.arange(train_size)
        perm = perm.astype(np.int32)
        
        np.random.shuffle(perm)  
        #perm = np.random.shuffle(perm)  错误，这样perm就是None了
        contents = X_train[perm]
        labels = y_train[perm]
         
        ##每一轮清零索引
        index_in_epoch = 0
        i=0
        while(i<train_size):
            index_in_epoch = i
            batch = get_next_batch(BS, index_in_epoch, contents, labels)

            _X_reshape = batch[0].reshape(-1,img_h,img_w,img_d)
            _y = batch[1]
            
            i = i + BS
            _, cost = sess.run([optimizer, cross_entropy], feed_dict={x:_X_reshape,y:_y,keep_prob:kb})
            
            step = i/BS
            if step%100==0:
                batch_acc, cost = sess.run([accuracy,cross_entropy], feed_dict={x:_X_reshape, y: _y, keep_prob:kb})
                print "Epoch:", e, "step:",step, " cost=", "{:.9f}".format(cost),  "acc=", "{:.9f}".format(batch_acc)

                 
     
      