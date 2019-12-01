# Kyle J. Cantrell & Craig D. Miller
# kjcantrell@wpi.edu & cmiller@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Depth Estimation from RGB Images

import numpy as np
#from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, Reshape
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
import deep_utils
import image_utils
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
from glob import glob
#import unet
import matplotlib.pyplot as plt
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models.segnet import resnet50_segnet
from segmentation_models import Unet
from losses import sil

start=time.time()
#Initialize tensorflow GPU settings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def model1():
	'''Define network model'''
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(480,640,3), activation='relu')) #X_train.shape[1], X_train.shape[2], X_train.shape[3]
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
	model.add(Dense(480*640,activation='relu')) #tanh, activation='softmax' #X_train.shape[1]*X_train.shape[2]
	model.compile(loss='mean_squared_error', optimizer='adam') #metrics=['mse']
	return model

def model2():
    model = Sequential()
    model.add(Convolution2D(30, 11, 11, border_mode='valid', input_shape=(480,640,3), activation='relu')) #X_train.shape[1], X_train.shape[2], X_train.shape[3]
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, 3, 3, activation='relu'))
    model.add(Convolution2D(10, 3, 3, activation='relu'))
    model.add(Convolution2D(5, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu',kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu',kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu',kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(480*640,activation='tanh',kernel_initializer='he_normal')) #, activation='softmax' #X_train.shape[1]*X_train.shape[2]
    model.compile(loss='mean_squared_error', optimizer='adam') #metrics=['mse']
    return model  
  
def rcnn():
   '''CNN First'''
   cnn = Sequential()
   cnn.add(Convolution2D(30, (5, 5),strides=(1,1), padding='valid', input_shape=(480,640,3), activation='relu'))
   cnn.add(MaxPooling2D(pool_size=(2, 2)))
   cnn.add(Convolution2D(15, (3, 3), activation='relu',strides=(1,1)))
   cnn.add(MaxPooling2D(pool_size=(2, 2)))
   cnn.add(Dropout(0.5))
   cnn.add(Flatten())
   cnn.add(Dense(128,activation='relu'))
   cnn.add(Dropout(0.5))
   #cnn.add(Reshape((91, 4620)))
   #cnn.add(LSTM(512,input_shape=(91,4620)))
   cnn.add(Dense(480*640,activation='linear'))
   cnn.compile(loss='mean_squared_error', optimizer=Adam(lr = 1e-4), metrics=['mse', 'mae', 'mape', 'cosine']) #,decay=0.01
   return cnn

pickle_files_folderpath=r"G:\Documents\NYU Depth Dataset\nyu_data\pickled"
X_files=glob(pickle_files_folderpath+'\\X_*')
y_files=glob(pickle_files_folderpath+'\\y_*')

X_test_files=r"G:\Documents\NYU Depth Dataset\nyu_data\pickled_test\X_0.p"
y_test_files=r"G:\Documents\NYU Depth Dataset\nyu_data\pickled_test\y_0.p"
X_test,y_test=deep_utils.load_pickle_files(X_test_files, y_test_files)
X_test,y_test=deep_utils.simul_shuffle(X_test,y_test) 
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]).astype(np.uint8)
y_test = y_test.reshape((y_test.shape[0],1,-1)).astype(np.uint8)
y_test = y_test.squeeze()

X_test=np.divide(X_test,255).astype(np.float16)
y_test=np.divide(y_test,255).astype(np.float16)
    
num_training_batches=len(X_files)
history=[]

for i in range(num_training_batches):  
    print('Batch '+str(i)+': '+'Loading data')
    X_train,y_train=deep_utils.load_pickle_files(X_files[i], y_files[i])
    X_train,y_train=deep_utils.simul_shuffle(X_train,y_train)
    print('Batch '+str(i)+': '+'Splitting data')
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed,shuffle=True)

    #Clear variables for memory
#    X=None
#    y=None
    
    print('Batch '+str(i)+': '+'Reshaping data') #[samples][width][height][pixels]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype(np.uint8)
    y_train = y_train.reshape((y_train.shape[0],1,-1)).astype(np.uint8)
    y_train = y_train.squeeze()
         
    print('Batch '+str(i)+': '+'Normalizing data')
    # normalize inputs and outputs from 0-255 to 0-1
    X_train=np.divide(X_train,255).astype(np.float16)   
    y_train=np.divide(y_train,255).astype(np.float16)
    
    if i==0:
        print('Building model')
        #model=pspnet_50_ADE_20K()
        #premodel = resnet50_segnet(n_classes=51 ,  input_height=480, input_width=640)
        premodel = Unet('resnet34', input_shape=(480, 640, 3), encoder_weights='imagenet',encoder_freeze = False)
        x=premodel.layers[-3].output
#        conv = Conv2D(1, 1, activation = 'linear',kernel_regularizer=l2(0.01),name='convolutional')(x)
#        reshape=Reshape((480*640,))(conv)
        flatten=Flatten()(x)
        dropout1=Dropout(0.5)(flatten)
        dense1=Dense(1,activation='relu',kernel_regularizer=l2(0.01))(dropout1)
        dense2=Dense(307200,activation='linear',kernel_regularizer=l2(0.01))(dense1)
#        reshape=Reshape((-1,480,640,1))(dense)
        model = Model(input=premodel.input, output=dense2)
        model.compile(loss=sil, optimizer=Adam(lr = 1e-5),metrics=['mean_squared_error']) #1e-4

    print('Batch '+str(i)+': '+'Fitting model')
    #checkpointer = ModelCheckpoint(filepath='best_checkpoint_weights.hdf5', verbose=1, save_best_only=True)
    history.append(model.fit(X_train, y_train,validation_data=(X_test, y_test), 
                             epochs=5, batch_size=1, verbose=2,)) #callbacks=[checkpointer]))
    
    #deep_utils.plot_accuracy(history)
    plt.figure()
    deep_utils.plot_loss(history[i])
#    deep_utils.plot_mse(history[i])
    
    image_utils.image_from_np(np.multiply(X_test[0],255).astype(np.uint8))
    test_image=X_test[0].reshape(1,X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2])
    y_est=model.predict(test_image)
    y_est=y_est.reshape((X_train.shape[1],X_train.shape[2]))*255 #De-normalize for depth viewing
    print('Sample image, Batch '+str(i))
    image_utils.heatmap(y_est)

print(model.summary())
finish=time.time()
elapsed=finish-start
print('Runtime :'+str(elapsed)+' seconds')
# from keras.utils import plot_model
# plot_model( m , show_shapes=True , to_file='model.png')
    
deep_utils.save_model(model,serialize_type='yaml',model_name='depth_estimation_cnn_nyu_model')

deep_utils.plot_full_val_loss(history)

#Show Image and predicted results
for i in [0,1,2]:  
    image_utils.image_from_np(np.multiply(X_test[i],255).astype(np.uint8))  #De-normalize for viewing
    test_image=X_test[i].reshape(1,X_test[i].shape[0],X_test[i].shape[1],X_test[i].shape[2])
    test_depth=y_test[i].reshape(480,640)
    #image_utils.image_from_np(test_depth)
    #plt.figure()
   # plt.imshow(test_depth, cmap='gray', interpolation='nearest')
    y_est=model.predict(test_image)
    y_est=y_est.reshape((X_train.shape[1],X_train.shape[2]))*255 #De-normalize for depth viewing
    print('Sample image, X_test['+str(i)+']:')
    image_utils.heatmap(y_est)

#Test new image
test_image=image_utils.rgb_read(r"C:\Users\Craig\Desktop\test5.png") #640x480
test_image=test_image.reshape(1,X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2])
test_image=np.divide(test_image,255).astype(np.float16)
y_est=model.predict(test_image)
y_est=y_est.reshape((X_train.shape[1],X_train.shape[2]))*255 #De-normalize for depth viewing
print('New Test Image:')
image_utils.heatmap(y_est)