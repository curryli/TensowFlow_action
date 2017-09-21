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
 
 
from keras import regularizers

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score
 
#导入随机森林算法库
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#from keras.datasets import mnist  
#from keras.utils.visualize_util import plot  
import sys  

import os                          #python miscellaneous OS system tool
#os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
 
print "start"
################################################################
df_All = pd.read_csv("idx_new_08_del.csv", sep=',') 
df_All = shuffle(df_All) 
print df_All.shape[1]
 
df_dummies =  df_All[["settle_tp","settle_cycle","block_id","trans_fwd_st","sms_dms_conv_in","cross_dist_in","tfr_in_in","trans_md","source_region_cd","dest_region_cd","cups_card_in","cups_sig_card_in","card_class","card_attr","acq_ins_tp","fwd_ins_tp","rcv_ins_tp","iss_ins_tp","acpt_ins_tp","resp_cd1","resp_cd3","cu_trans_st","sti_takeout_in","trans_id","trans_tp","trans_chnl","card_media","trans_curr_cd","conn_md","msg_tp","msg_tp_conv","trans_proc_cd","trans_proc_cd_conv","mchnt_tp","pos_entry_md_cd","pos_cond_cd","pos_cond_cd_conv","term_tp","rsn_cd","addn_pos_inf","iss_ds_settle_in","acq_ds_settle_in","upd_in","pri_cycle_no","disc_in","fwd_settle_conv_rt","fwd_settle_curr_cd","rcv_settle_curr_cd","acq_ins_id_cd_BK","acq_ins_id_cd_RG","fwd_ins_id_cd_BK","fwd_ins_id_cd_RG","rcv_ins_id_cd_BK","rcv_ins_id_cd_RG","iss_ins_id_cd_BK","iss_ins_id_cd_RG","acpt_ins_id_cd_BK","acpt_ins_id_cd_RG","settle_fwd_ins_id_cd_BK","settle_fwd_ins_id_cd_RG","settle_rcv_ins_id_cd_BK","settle_rcv_ins_id_cd_RG"]]
 

df_X = pd.concat([df_All[["tfr_dt_tm","day_week","hour","trans_at","total_disc_at"]],df_dummies], axis=1).as_matrix()
#df_X = df_All[["hour","trans_at","total_disc_at"]]
print " df_X.shape", df_X.shape

df_y = df_All["label"].as_matrix()

x_train, x_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)

sc = MinMaxScaler()   #这里如果用StandardScaler，那么会出现负数，导致算出来的KL距离 loss会有负数，

#print X_train.loc[:1]

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


print " x_train.shape", x_train.shape,  "x_train.shape", x_test.shape 
##################################################################


  
batch_size = 100
original_dim = 67  
latent_dim = 2  
intermediate_dim = 64  
nb_epoch = 20  
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
    return K.mean(xent_loss + kl_loss) 
  
vae = Model(x, x_decoded_mean)  
vae.compile(optimizer='rmsprop', loss=vae_loss)  
  


vae.fit(x_train, x_train,  
        shuffle=True,  
        epochs=nb_epoch,  
        verbose=2,   
        batch_size=batch_size,
        validation_data=(x_test, x_test))  
  
# build a model to project inputs on the latent space  
encoder = Model(x, z_mean)  
 
  
# build a digit generator that can sample from the learned distribution  使用vae  跟普通自动编码器不一样，我们这里只需要 剪掉编码器部分，直接把正态分布样本送入解码器即可
decoder_input = Input(shape=(latent_dim,))  
_h_decoded = decoder_h(decoder_input)  
_x_decoded_mean = decoder_mean(_h_decoded)  
generator = Model(decoder_input, _x_decoded_mean)  
 