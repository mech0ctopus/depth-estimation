# -*- coding: utf-8 -*-
"""
Final Models.
"""
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, LSTM, Input
from tensorflow.keras.layers import Flatten, Reshape, Concatenate
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
import segmentation_models

# from models.losses import sil
# from models.test_loss import depth_loss_function

segmentation_models.set_framework('tf.keras')

def cnn(input_shape=(480,640,3)):
	'''Define CNN model'''
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid',input_shape=input_shape, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	#model.add(Flatten())	
	model.add(Dense(480*640,activation='linear'))
	#model.compile(loss='mean_squared_error', optimizer=Adam(),metrics=['mse','msle']) 
	return model

def pretrained_unet():
    '''Define pretrained U-Net model.'''
    #Load unet with resnet34 backbone.  Freeze imagenet weights for encoder
    premodel = segmentation_models.Unet('resnet34', input_shape=(480, 640, 3), encoder_weights='imagenet',encoder_freeze = True)
    #Get final conv. output and skip sigmoid activation layer
    x=premodel.layers[-2].output 
    reshape=Reshape((307200,))(x)
    model = Model(inputs=premodel.input, outputs=reshape)
    #model.compile(loss='mean_squared_error', optimizer=Adam(),metrics=['mse','msle']) #lr=1e-5, lr=5e-3
    #model.compile(loss=depth_loss_function, optimizer=Adam(),metrics=['mse','msle'])
    return model

def pretrained_unet_cnn():
    '''Define pretrained U-Net with CNN model.'''
    unet_cnn=Sequential()
    #Load unet with resnet34 backbone.  Freeze imagenet weights for encoder
    premodel = segmentation_models.Unet('resnet34', input_shape=(480, 640, 3), encoder_weights='imagenet',encoder_freeze = True)
    #Get final conv. output and keep sigmoid activation layer
    premodel=Model(inputs=premodel.input,outputs=premodel.layers[-1].output)  
    unet_cnn.add(premodel)
    unet_cnn.add(cnn(input_shape=(480,640,1)))

    #unet_cnn.compile(loss='mean_squared_error', optimizer=Adam()) 
    return unet_cnn

def wnet():
    #Load unet with resnet34 backbone.
    wnet=Sequential()
    firstU = segmentation_models.Unet('resnet34', input_shape=(480, 640, 3), encoder_weights='imagenet',encoder_freeze = True)
    secondU = segmentation_models.Unet('resnet34', input_shape=(480, 640, 1), encoder_weights=None)
    #Get final conv. output and keep sigmoid activation layer
    firstU = Model(inputs=firstU.input, outputs=firstU.layers[-1].output)
    #Get final conv. output and skip sigmoid activation layer
    secondU=Model(inputs=secondU.input, outputs=secondU.layers[-2].output)
#    for layer in firstU.layers:
#        layer.trainable = False
    for layer in secondU.layers:
        layer.trainable = True
        
    wnet.add(firstU)
    wnet.add(secondU)
    wnet.add(Reshape((307200,)))
    
    # Make sure that the pre-trained firstU layers are not trainable
#    wnet.layers[0].trainable=False
#    wnet.layers[1].trainable=True
#    wnet.layers[2].trainable=True
    #wnet.compile(loss='mean_squared_error', optimizer=Adam())
    #wnet.summary()
    return wnet

def wnet_connected():   
    #Load unet with resnet34 backbone. (densenet201,resnet34,vgg16,resnet18,resnet152)
    firstU = segmentation_models.Unet('resnet34', input_shape=(480, 640, 3), encoder_weights='imagenet',encoder_freeze = True)
    secondU = segmentation_models.Unet('resnet34', input_shape=(480, 640, 4), encoder_weights=None)
    #Get final conv. output and keep sigmoid activation layer
    firstU = Model(inputs=firstU.input, outputs=firstU.layers[-1].output)
    #Get final conv. output and skip sigmoid activation layer
    secondU=Model(inputs=secondU.input, outputs=secondU.layers[-2].output) 
#    for layer in firstU.layers:
#        layer.trainable = False
    for layer in secondU.layers:
        layer.trainable = True
    
    inputs = Input((480,640,3))
    m1=firstU(inputs)
    merged=Concatenate()([inputs,m1])
    reshape1=Reshape((480,640,4))(merged)
    m2=secondU(reshape1)
    reshape2=Reshape((307200,))(m2)
    
    wnet_c=Model(inputs=inputs,outputs=reshape2)
    
    # Make sure that the pre-trained firstU layers are not trainable
#    for layer in wnet_c.layers:
#        print(layer)
#    wnet_c.layers[0].trainable=False #Input
#    wnet_c.layers[1].trainable=False #First U
    wnet_c.layers[2].trainable=True #Concat
    wnet_c.layers[3].trainable=True #Reshape
    wnet_c.layers[4].trainable=True #Second U
    wnet_c.layers[5].trainable=True #Reshape
    #wnet_c.compile(loss='mean_squared_error', optimizer=Adam(),metrics=['mse','msle']) #lr=1e-5, lr=0.00001
    #wnet_c.summary()

    return wnet_c

def rcnn_640_480(input_shape=(480,640,3)):
   '''RCNN: CNN First'''
   rcnn = Sequential()
   rcnn.add(Convolution2D(30, (10, 10),strides=(1,1), padding='valid', input_shape=input_shape, activation='relu'))
   rcnn.add(MaxPooling2D(pool_size=(4, 4)))
   rcnn.add(Dropout(0.5))
   rcnn.add(Convolution2D(15, (6, 6), activation='relu',strides=(1,1)))
   rcnn.add(MaxPooling2D(pool_size=(4, 4)))
   rcnn.add(Dropout(0.5))
   rcnn.add(Flatten())
   rcnn.add(Reshape((28, 38*15)))
   rcnn.add(LSTM(512,input_shape=(28,38*15),return_sequences=True))
   rcnn.add(Dense(512,activation='relu'))
   rcnn.add(LSTM(512))
   rcnn.add(Dense(128,activation='relu'))
   rcnn.add(Dense(640*480,activation='linear'))
   #rcnn.compile(loss='mean_squared_error', optimizer=Adam())
   return rcnn

def pretrained_unet_rcnn():
    '''Define pretrained U-Net with RCNN model.'''
    unet_rcnn=Sequential()
    #Load unet with resnet34 backbone.  Freeze imagenet weights for encoder
    premodel = segmentation_models.Unet('resnet34', input_shape=(480, 640, 3), encoder_weights='imagenet',encoder_freeze = True)
    #Get final conv. output and keep sigmoid activation layer
    premodel=Model(inputs=premodel.input,outputs=premodel.layers[-1].output)  
    unet_rcnn.add(premodel)
    unet_rcnn.add(rcnn_640_480(input_shape=(480,640,1)))

    #unet_rcnn.compile(loss='mean_squared_error', optimizer=Adam()) 
    return unet_rcnn