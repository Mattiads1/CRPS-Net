import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

def dense_net(shape_in = (40,), M = 100):
 
    in_put=Input(shape=shape_in, name='input')
    x = Dense(units=400, activation='swish', name='hidden_0')(in_put)
    x = Dense(units=600, activation='elu', name='hidden_1')(x)
    x = Dense(units=400, activation='gelu', name='hidden_2')(x)
   
    out = Dense(M,activation='relu',name='ensemble_output')(x)

    # out = tf.math.reduce_mean(x,axis=-1,keepdims=True)

    model=Model(inputs=in_put,outputs=out)
    
    return model

######################################################################################

class MYPadding1D(tf.keras.layers.Layer):
    def __init__(self, pad, **kwargs):
        super(MYPadding1D, self).__init__(**kwargs)
        self.pad = pad

    def MY_padding_border(self,inpt, pad):
    ### Only work for pad > 0 ###
        L = inpt[:,:pad[0],:]
        R = inpt[:,-pad[1]:,:]
        inpt_pad = tf.concat([L, inpt, R], axis=1)
        return inpt_pad

    def call(self, inputs):
        return self.MY_padding_border(inputs, self.pad)

def conv_net(in_shape=(24,1), M = 100):

    in_l = tf.keras.layers.Input(in_shape)
    in_lp = MYPadding1D(pad = (2,2))(in_l)
    c1 = tf.keras.layers.Conv1D(200,kernel_size = 5,strides = 1,activation = 'relu',padding='valid')(in_lp)
    lstm1 = tf.keras.layers.LSTM(200,return_sequences=True)(c1)
    lstm1b = tf.keras.layers.LSTM(200,return_sequences=True,go_backwards=True)(c1)
    clstm = tf.keras.layers.Concatenate()([lstm1,lstm1b])
    clstm = MYPadding1D(pad = (2,2))(clstm)
    cout =  tf.keras.layers.Conv1D(150,kernel_size = 5,strides = 1,activation = 'relu',padding='valid')(clstm)
    cout =  MYPadding1D(pad = (1,1))(cout)
    out =  tf.keras.layers.Conv1D(M ,kernel_size = 3,strides = 1,activation = 'linear', padding='valid')(cout)
    
    return tf.keras.Model(in_l,out)





    





