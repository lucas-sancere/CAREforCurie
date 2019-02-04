#Implementation of 2D or 3D Unet, multi channel

from __future__ import print_function, unicode_literals, absolute_import, division
from keras.layers import UpSampling2D, UpSampling3D, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D

from keras.layers import Activation, Dropout, BatchNormalization
from keras.layers.merge import Concatenate


def Conv2D(n_filter, n1, n2,activation = 'relu', border_mode = 'same', dropout = 0.0, batch_norm = 'False', init = 'glorot_uniform', **kwargs ):
    
    
    def main_function(layer):
        if batch_norm:
            s = Conv2D(n_filter, (n1, n2), padding = border_mode,kernel_initializer = init, **kwargs)(layer)
            s = BatchNormalization()(s)
            s = Activation(activation)(s)
            
        else:
            
            s = Conv2D(n_filter, (n1, n2), padding = border_mode,  kernel_initializer = init, activation = activation, **kwargs)(layer)
       
       
        if dropout is not None and dropout > 0:
            
            s= Dropout(dropout)(s)
            
        return s
    
    
    return main_function 


def Conv3D(n_filter, n1, n2, n3, activation = 'relu', border_mode = 'same', dropout = 0.0, batch_norm = 'False', init = 'glorot_uniform', **kwargs):
    
    def main_function(layer):
        if batch_norm:
            s = Conv3D(n_filter, (n1, n2, n3), padding = border_mode, kernel_initializer = init, **kwargs)(layer)
            s = BatchNormalization()(s)
            s = Activation(activation)(s)
            
        else:
            
            s = Conv3D(n_filter, (n1,n2,n3), padding = border_mode, kernel_initializer = init, activation = activation, **kwargs  )(layer)
            
        if droupout is not None and dropout > 0:
            
            s = Dropout(dropout)(s)
            
        return s
    
    return main_function
    
