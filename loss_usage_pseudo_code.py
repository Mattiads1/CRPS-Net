import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import sys
sys.path.append('./')

import CRPSLoss as lf
import nn as net

import tensorflow as tf
keras = tf.keras

'''

    Import from tensorflow all you need to train your neural network
    Exemple:
        
'''
from tensorflow.keras.optimizers import Adam 

######################################################################

'''

    Create your personal multi-output neural network
    Exemple:

'''

model = net.dense_net()

model.summary()

######################################################################

'''

    Set the CRPS Loss and compile your nodel.
    Chose if run in paramteric or non-parametric set-up

'''

lossfunc = lf.CRPSLoss(parametric = False)

model.compile(optimizer = Adam(), loss = lossfunc )

model.fit('x_train', 'y_train',
          batch_size = 256, epochs = 500,
          callbacks = ['your personal callbacks'], 
          validation_data = ('x_val', 'y_val'))


