# -*- coding: utf-8 -*-
"""
unet

https://github.com/zhixuhao/unet/blob/master/model.py
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, Dense, Flatten, Reshape
from keras.optimizers import Adam
from keras.regularizers import l2

def unet(pretrained_weights = None,input_size = (480,640,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(pool2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(pool3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(pool4)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(conv6)

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(conv7)

    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(conv8)

    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(merge9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.01))(conv9)

    #flatten10 = Flatten()(conv9)   
    conv10 = Conv2D(1, 1, activation = 'linear',kernel_regularizer=l2(0.01))(conv9)
    reshape10=Reshape((480*640,))(conv10)
    
#    flatten10 = Flatten()(conv9)
#    dense10 = Dense(128, activation='relu',kernel_regularizer=l2(0.01))(flatten10)
#    dense11 = Dense(480*640, activation='linear',kernel_regularizer=l2(0.01))(dense10)

    model = Model(input = inputs, output = reshape10)

    model.compile(optimizer = Adam(), loss = 'mean_squared_error') #lr = 1e-4
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model