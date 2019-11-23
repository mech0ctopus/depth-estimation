# Kyle J. Cantrell & Craig D. Miller
# kjcantrell@wpi.edu & cmiller@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Depth Estimation from RGB Images

import numpy as np
#from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
import deep_utils
import image_utils
#from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
from glob import glob

start=time.time()
#Initialize tensorflow GPU settings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def larger_model():
	'''Define network model'''
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(375,1242,3), activation='relu')) #X_train.shape[1], X_train.shape[2], X_train.shape[3]
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
#	model.add(Dense(128, activation='relu',kernel_initializer='he_normal'))
#	model.add(Dropout(0.5))
#	model.add(Dense(128, activation='relu',kernel_initializer='he_normal'))
#	model.add(Dropout(0.5))
#	model.add(Dense(128, activation='relu',kernel_initializer='he_normal'))
#	model.add(Dropout(0.5))
#	model.add(Dense(64, activation='relu',kernel_initializer='he_normal'))
#	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu',kernel_initializer='he_normal'))
	model.add(Dropout(0.5))
	#model.add(Flatten())	
	model.add(Dense(375*1242,activation='tanh')) #, activation='softmax' #X_train.shape[1]*X_train.shape[2]
	model.compile(loss='mean_squared_error', optimizer='adam') #metrics=['mse']
	return model

val_pickle_files_folderpath=r"G:\Documents\KITTI\pickled_KITTI\validation"
X_test_files=glob(val_pickle_files_folderpath+'\\X_*')
y_test_files=glob(val_pickle_files_folderpath+'\\y_*')

#Build X_test & y_test arrays from pickle files
for i in range(len(y_test_files)):
    if i>0:
        X_test_i,y_test_i=deep_utils.load_pickle_files(X_test_files[i], y_test_files[i])
        X_test=np.concatenate((X_test,X_test_i))
        y_test=np.concatenate((y_test,y_test_i))
        X_test_i=None
        y_test_i=None
    else:
        X_test,y_test=deep_utils.load_pickle_files(X_test_files[i], y_test_files[i])
    
pickle_files_folderpath=r"G:\Documents\KITTI\pickled_KITTI"
X_files=glob(pickle_files_folderpath+'\\X_*')
y_files=glob(pickle_files_folderpath+'\\y_*')

num_training_batches=len(X_files)
history=[]

for i in range(num_training_batches):  
    print('Batch '+str(i)+': '+'Loading data')
    X_train,y_train=deep_utils.load_pickle_files(X_files[i], y_files[i])
    X_train,y_train=deep_utils.simul_shuffle(X_train,y_train)
    
    print('Batch '+str(i)+': '+'Splitting data')
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed,shuffle=True)
    
    #Clear variables for memory
#    X=None
#    y=None
    
    print('Batch '+str(i)+': '+'Reshaping data') #[samples][width][height][pixels]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype(np.uint8)
    y_train = y_train.reshape((y_train.shape[0],1,-1)).astype(np.uint8)
    y_train = y_train.squeeze()
    
    if i==0:
        print('Reshaping test data')
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]).astype(np.uint8)
        y_test = y_test.reshape((y_test.shape[0],1,-1)).astype(np.uint8)
        y_test = y_test.squeeze()
        
        print('Normalizing test data')
        X_test=np.divide(X_test,255).astype(np.float16)
        y_test=np.divide(y_test,255).astype(np.float16)
         
    print('Batch '+str(i)+': '+'Normalizing data')
    # normalize inputs and outputs from 0-255 to 0-1
    X_train=np.divide(X_train,255).astype(np.float16)   
    y_train=np.divide(y_train,255).astype(np.float16)
    
    if i==0:
        print('Building model')
        model = larger_model()

    print('Batch '+str(i)+': '+'Fitting model')
    #checkpointer = ModelCheckpoint(filepath='best_checkpoint_weights.hdf5', verbose=1, save_best_only=True)
    history.append(model.fit(X_train, y_train,validation_data=(X_test, y_test), 
                             epochs=5, batch_size=16, verbose=2,)) #callbacks=[checkpointer]))
    
    #deep_utils.plot_accuracy(history)
    deep_utils.plot_loss(history[i])
    #deep_utils.plot_mse(history)

print(model.summary())
finish=time.time()
elapsed=finish-start
print('Runtime :'+str(elapsed)+' seconds')

deep_utils.save_model(model,serialize_type='yaml',model_name='depth_estimation_cnn_model')

deep_utils.plot_full_val_loss(history)
        
#Show Image and predicted results
for i in [0,25,50]:  
    image_utils.image_from_np(np.multiply(X_test[i],255).astype(np.uint8))  #De-normalize for viewing
    test_image=X_test[i].reshape(1,X_test[i].shape[0],X_test[i].shape[1],X_test[i].shape[2])
    y_est=model.predict(test_image)
    y_est=y_est.reshape((X_train.shape[1],X_train.shape[2]))*255 #De-normalize for depth viewing
    print('Sample image, X_test['+str(i)+']:')
    image_utils.heatmap(y_est)

#Test new image
test_image=image_utils.rgb_read(r"G:\Pictures\Pictures\resize2_CDM_05082017.jpg")
test_image=X_test[i].reshape(1,X_test[i].shape[0],X_test[i].shape[1],X_test[i].shape[2])
y_est=model.predict(test_image)
y_est=y_est.reshape((X_train.shape[1],X_train.shape[2]))*255 #De-normalize for depth viewing
print('New Test Image:')
image_utils.heatmap(y_est)