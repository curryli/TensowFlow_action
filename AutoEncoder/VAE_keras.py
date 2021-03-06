# -*- coding: utf-8 -*-
'''''This script demonstrates how to build a variational autoencoder with Keras. 
https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py 
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114 
'''  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import norm  
  
from keras.layers import Input, Dense, Lambda  
from keras.models import Model  
from keras import backend as K  
from keras import objectives  
#from keras.datasets import mnist  
#from keras.utils.visualize_util import plot  
import sys  

import os                          #python miscellaneous OS system tool
#os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
 
batch_size = 100  
original_dim = 784   #28*28  
latent_dim = 2  
intermediate_dim = 256  
nb_epoch = 50  
epsilon_std = 1.0  
  
#my tips:encoding  
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
  
#my tips:Gauss sampling,sample Z  
def sampling(args):   
    z_mean, z_log_var = args  
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,  
                              stddev=epsilon_std)  
    return z_mean + K.exp(z_log_var / 2) * epsilon  
  
# note that "output_shape" isn't necessary with the TensorFlow backend  
# my tips:get sample z(encoded)  
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])   #图中z
  
# we instantiate these layers separately so as to reuse them later  
decoder_h = Dense(intermediate_dim, activation='relu')  
decoder_mean = Dense(original_dim, activation='sigmoid')  
h_decoded = decoder_h(z)   #图中f(z)
x_decoded_mean = decoder_mean(h_decoded)    #图中最后一个x
  
#my tips:loss(restruct X)+KL   目标函数有两项   
def vae_loss(x, x_decoded_mean):  
      #my tips:logloss   一项与自动编码器相同，要求从f(z)出来的样本重构原来的输入样本   由重构x与输入x均方差或逐点的交叉熵衡量
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)  
    #my tips:see paper's appendix B    另一项 要求经过Q(z|x)估计出的隐变量分布接近于标准正态分布    由衡量两个分布的相似度，当然是大名鼎鼎的KL距离。二是近似后验与真实后验的KL散度，至于KL散度为何简化成代码中的形式，看论文《Auto-Encoding Variational Bayes》中的附录B有证明。
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)  
    return xent_loss + kl_loss  
  
vae = Model(x, x_decoded_mean)  
vae.compile(optimizer='rmsprop', loss=vae_loss)  
  
# train the VAE on MNIST digits  
def load_data(path='mnist.npz'):
    path = 'mnist.npz'
    f = np.load(path)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
                                                          


(x_train, y_train), (x_test, y_test) = load_data()
  
x_train = x_train.astype('float32') / 255.  
x_test = x_test.astype('float32') / 255.  
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))  
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))  
  
vae.fit(x_train, x_train,  
        shuffle=True,  
        nb_epoch=nb_epoch,  
        verbose=2,  
        batch_size=batch_size,  
        validation_data=(x_test, x_test))  
  
# build a model to project inputs on the latent space  
encoder = Model(x, z_mean)  
  
# display a 2D plot of the digit classes in the latent space  
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  
plt.figure(figsize=(6, 6))  
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)  
plt.colorbar()  
plt.show()  
  
# build a digit generator that can sample from the learned distribution  使用vae  跟普通自动编码器不一样，我们这里只需要 剪掉编码器部分，直接把正态分布样本送入解码器即可
decoder_input = Input(shape=(latent_dim,))  
_h_decoded = decoder_h(decoder_input)  
_x_decoded_mean = decoder_mean(_h_decoded)  
generator = Model(decoder_input, _x_decoded_mean)  
  
# display a 2D manifold of the digits  
n = 15  # figure with 15x15 digits  
digit_size = 28  
figure = np.zeros((digit_size * n, digit_size * n))  
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian  
# to produce values of the latent variables z, since the prior of the latent space is Gaussian  
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))  
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))  
  
for i, yi in enumerate(grid_x):  
    for j, xi in enumerate(grid_y):  
        z_sample = np.array([[xi, yi]])  
        x_decoded = generator.predict(z_sample)  
        digit = x_decoded[0].reshape(digit_size, digit_size)  
        figure[i * digit_size: (i + 1) * digit_size,  
               j * digit_size: (j + 1) * digit_size] = digit  
  
plt.figure(figsize=(10, 10))  
plt.imshow(figure, cmap='Greys_r')  
plt.show()  
  
#plot(vae,to_file='variational_autoencoder_vae.png',show_shapes=True)  
#plot(encoder,to_file='variational_autoencoder_encoder.png',show_shapes=True)  
#plot(generator,to_file='variational_autoencoder_generator.png',show_shapes=True)  
  
 