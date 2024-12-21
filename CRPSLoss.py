import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import tensorflow_probability as tfp
import numpy as np


class CRPSLoss(tf.keras.losses.Loss):
    def __init__(self, name='CRPSLoss', reduction='sum_over_batch_size',parametric = False,**kwargs):
        super().__init__(name=name,reduction = reduction, **kwargs)
        self.parametric = parametric

    def call(self,y_true,y_pred):

        y_pred = tf.cast(y_pred,tf.float32)
        y_true = tf.cast(y_true,tf.float32)
        
        
        if self.parametric:
        
           mu  = tf.math.reduce_mean(y_pred,axis=-1,keepdims = True)
           sigma = tf.math.reduce_std(y_pred,axis=-1,keepdims =  True)  
           pi = tf.cast(np.pi,tf.float32)
           dist = tfp.distributions.Normal(0.,1.)
           z = (y_true-mu)/sigma
           crps_loss = sigma * ( z * (2 * dist.cdf(z)-1) + 2 * dist.prob(z) - 1/tf.math.sqrt(pi) )
           
        else:
            
            crps1 = tf.math.reduce_mean(tf.math.abs(y_pred - y_true),axis=-1)
            crps2 = tf.zeros_like(crps1)  
            M = tf.shape(y_pred)[-1]
            for i in range(M):
                crps2 +=  tf.math.reduce_sum((tf.math.abs(tf.expand_dims(y_pred[:,i],-1) - y_pred)),axis=-1)
                
            M = tf.cast(M,tf.float32)
            crps_loss = crps1 - 1/(2*M**2)*crps2

        return crps_loss

