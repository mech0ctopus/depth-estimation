# Kyle Cantrell & Craig Miller
# cmiller@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Depth Estimation from RGB Images

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import deep_utils
import image_depth
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time

#import random

start=time.time()
#Initialize tensorflow GPU settings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

print('Loading data')
#Load data from pickle files
X,y=deep_utils.load_pickle_files(r"X.p", r"y.p")
X,y=deep_utils.simul_shuffle(X,y)

print('Splitting data')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed,shuffle=True)

#Clear variables for memory
X=None
y=None

print('Reshaping data')
## reshape to be [samples][width][height][pixels]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype(np.uint8)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]).astype(np.uint8)
y_train=y_train.reshape((X_train.shape[0],1,-1)).astype(np.uint8)
y_test=y_test.reshape((X_test.shape[0],1,-1)).astype(np.uint8)

y_train = y_train.squeeze()
y_test = y_test.squeeze()
print('Normalizing data')
# normalize inputs and outputs from 0-255 to 0-1
X_train=np.divide(X_train,255).astype(np.float16)
X_test=np.divide(X_test,255).astype(np.float16)
y_train=np.divide(y_train,255).astype(np.float16)
y_test=np.divide(y_test,255).astype(np.float16)

def larger_model():
	'''Define network model'''
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
#	model.add(Dense(128, activation='relu',init='he_normal'))
#	model.add(Dropout(0.5))
#	model.add(Dense(128, activation='relu',init='he_normal'))
#	model.add(Dropout(0.5))
#	model.add(Dense(128, activation='relu',init='he_normal'))
#	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu',init='he_normal'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu',init='he_normal'))
	model.add(Dropout(0.5))
	#model.add(Flatten())	
	model.add(Dense(1242*375,activation='tanh')) #, activation='softmax'
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model

print('Building model')
# build the model
model = larger_model()
print(model.summary())
print('Fitting model')
# Fit the model
history=model.fit(X_train, y_train,validation_data=(X_test, y_test), nb_epoch=5, batch_size=16, verbose=2)

finish=time.time()
elapsed=finish-start
print('Runtime :'+str(elapsed)+' seconds')

deep_utils.plot_accuracy(history)
deep_utils.plot_loss(history)

#Show Image and predicted results
for i in [0,25,50]:  
    image_depth.image_from_np(np.multiply(X_test[i],255).astype(np.uint8))  #De-normalize for viewing
    test_image=X_test[i].reshape(1,X_test[i].shape[0],X_test[i].shape[1],X_test[i].shape[2])
    y_est=model.predict(test_image)
    y_est=y_est.reshape((X_train.shape[1],X_train.shape[2]))*255 #De-normalize for depth viewing
    image_depth.heatmap(y_est)

#Test new image
test_image=image_depth.rgb_read(r"G:\Pictures\Pictures\resize2_CDM_05082017.jpg")
test_image=X_test[i].reshape(1,X_test[i].shape[0],X_test[i].shape[1],X_test[i].shape[2])
y_est=model.predict(test_image)
y_est=y_est.reshape((X_train.shape[1],X_train.shape[2]))*255 #De-normalize for depth viewing
image_depth.heatmap(y_est)

#deep_utils.save_model(model,serialize_type='yaml',model_name='depth_estimation_cnn_model')