
from keras.layers import Dropout, Activation, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, SeparableConv2D
from keras.layers.merge import Concatenate, Add
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
import keras.callbacks



def conv_block2(n_filter, n1, n2,
                activation="relu",
                border_mode="same",
                dropout=0.0,
                batch_norm=False,
                init="glorot_uniform",
                ):
    def _func(lay):
        s = Conv2D(n_filter, (n1, n2), padding=border_mode, kernel_initializer=init)(lay)
        if batch_norm:
            s = BatchNormalization()(s)
        s = Activation(activation)(s)
        if dropout > 0:
            s = Dropout(dropout)(s)
        return s

    return _func



def unet_block(n_depth=2, n_filter_base=16, n_row=3, n_col=3, n_conv_per_depth=2,
               activation="relu",
               batch_norm=False,
               dropout=0.0,
               last_activation=None):
    """"""

    if last_activation is None:
        last_activation = activation

    if K.image_dim_ordering() == "tf":
        channel_axis = -1
    else:
        channel_axis = 1


    def _func(input):
        skip_layers = []
        layer = input

        # down ...
        for n in range(n_depth):
            for i in range(n_conv_per_depth):
                layer = conv_block2(n_filter_base * 2 ** n, n_row, n_col,
                                    dropout=dropout,
                                    activation=activation,
                                    batch_norm=batch_norm)(layer)
            skip_layers.append(layer)
            layer = MaxPooling2D((2, 2))(layer)


        # middle
        for i in range(n_conv_per_depth - 1):
            layer = conv_block2(n_filter_base * 2 ** n_depth, n_row, n_col,
                                dropout=dropout,
                                activation=activation,
                                batch_norm=batch_norm)(layer)

        layer = conv_block2(n_filter_base * 2 ** (n_depth - 1), n_row, n_col,
                            dropout=dropout,
                            activation=activation,
                            batch_norm=batch_norm)(layer)

        # ...and up with skip layers
        for n in reversed(range(n_depth)):
            layer = Concatenate(axis = channel_axis)([UpSampling2D((2, 2))(layer), skip_layers[n]])
            for i in range(n_conv_per_depth - 1):
                layer = conv_block2(n_filter_base * 2 ** n, n_row, n_col,
                                    dropout=dropout,
                                    activation=activation,
                                    batch_norm=batch_norm)(layer)

            layer = conv_block2(n_filter_base * 2 ** max(0, n - 1), n_row, n_col,
                                dropout=dropout,
                                activation=activation if n > 0 else last_activation,
                                batch_norm=batch_norm)(layer)

        return layer

    return _func


# def my_binary_crossentropy(weights =(1., 1.)):
#     def _func(y_true, y_pred):
#         return -(weights[0] * K.mean((1-y_true)*K.log((1-y_pred)+K.epsilon())) +
#                  weights[1] * K.mean(y_true*K.log(y_pred+K.epsilon())))
#     return _func


def my_binary_crossentropy(weights =(1., 1.)):
    def _func(y_true, y_pred):
        return -(weights[0] * K.mean((1-y_true)*K.log((1-y_pred)+K.epsilon())) +
                 weights[1] * K.mean(y_true*K.log(y_pred+K.epsilon())))
    return _func

def my_binary_crossentropy_mod(weights =(1., 1.)):
    def _func(y_true, y_pred):
        return -(weights[0] * K.mean(K.cast(K.greater(0.25, y_true),dtype='float32')*K.log((1-y_pred)+K.epsilon())) +
                 weights[1] * K.mean(K.cast(K.greater(y_true,0.25),dtype='float32')*K.log(y_pred +K.epsilon())))
    return _func

def build_model_unet(input_shape, dropout=.2):
    input = Input(input_shape)

    unet = unet_block(2, 8, 3, 3, activation="relu")(input)
    final = Conv2D(1, (1, 1), activation='sigmoid')(unet)

    model = Model(inputs=input, outputs=final)

    return model

def acc1(y_true, y_pred):
   
   
    nom = K.mean(K.cast(K.cast(K.equal(K.round(y_pred),y_true), dtype='float32')*K.cast(K.equal(y_true,1),dtype='float32'),dtype='float32'))
    denom = K.mean(y_true)
#     nom = K.cast(nom, dtype='float32')
#     denom =  K.cast(denom, dtype='float32')
    return nom/denom
#     return K.shape(y_true)[0]



def acc0(y_true, y_pred):
    
    nom = K.mean(K.cast(K.cast(K.equal(K.round(y_pred),y_true), dtype='float32')*K.cast(K.equal(y_true,0),dtype='float32'),dtype='float32'))
    denom = K.mean(K.cast(K.equal(y_true,0),dtype='float32'))
#     nom = K.cast(nom, dtype='float32')
#     denom =  K.cast(denom, dtype='float32')
    return nom/denom

def acc1_mod(y_true, y_pred):
   
   
    nom = K.mean(K.cast(K.cast(K.greater(0.05,K.abs(y_pred-y_true)), dtype='float32')*K.cast(K.greater(y_true,0.2),dtype='float32'),dtype='float32'))
    denom = K.mean(K.cast(K.greater(y_true,0.2),dtype='float32'))
#     nom = K.cast(nom, dtype='float32')
#     denom =  K.cast(denom, dtype='float32')
    return nom/denom
#     return K.shape(y_true)[0]



def acc0_mod(y_true, y_pred):
    
    nom = K.mean(K.cast(K.cast(K.equal(K.round(y_pred),y_true), dtype='float32')*K.cast(K.equal(y_true,0),dtype='float32'),dtype='float32'))
    denom = K.mean(K.cast(K.equal(y_true,0),dtype='float32'))
#     nom = K.cast(nom, dtype='float32')
#     denom =  K.cast(denom, dtype='float32')
    return nom/denom