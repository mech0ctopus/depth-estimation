# Kyle J. Cantrell & Craig D. Miller
# kjcantrell@wpi.edu & cmiller@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Depth Estimation from RGB Images

import numpy as np
# import tensorflow as tf
import time
from glob import glob
# import matplotlib.pyplot as plt
from utils import deep_utils
# from utils import image_utils
from models import models
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
import datetime
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def _batchGenerator(X_files,y_files,batchSize):
    """
    Yield X and Y data when the batch is filled.
    """
    while True:
        for i in range(len(X_files)):
            #Load data
            X_train,y_train=deep_utils.load_pickle_files(X_files[i], y_files[i])
            X_train,y_train=deep_utils.simul_shuffle(X_train,y_train)
            
            #Reshape [samples][width][height][pixels]
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype(np.uint8)
            y_train = y_train.reshape((y_train.shape[0],1,-1)).astype(np.uint8)
            y_train = y_train.squeeze()
                 
            # normalize inputs and outputs from 0-255 to 0-1
            X_train=np.divide(X_train,255).astype(np.float16)   
            y_train=np.divide(y_train,255).astype(np.float16)
  
            j=0
            while j<X_train.shape[0]:
                x=X_train[j:j+1]
                y=y_train[j:j+1]
                j+=batchSize
                yield x,y
            
def main(model=models.wnet_connected,num_epochs=5,batch_size=2):
    '''Trains depth estimation model.'''
          
    #Load training data
    pickle_files_folderpath=r"G:\Documents\NYU Depth Dataset\nyu_data\pickled_colorized"
    X_files=glob(pickle_files_folderpath+'\\X_*')
    y_files=glob(pickle_files_folderpath+'\\y_*')
    X_files,y_files=deep_utils.simul_shuffle(X_files,y_files)
    
    #Load testing data
    X_test_files=r"G:\Documents\NYU Depth Dataset\nyu_data\pickled_test\X_40.p"
    y_test_files=r"G:\Documents\NYU Depth Dataset\nyu_data\pickled_test\y_40.p"
    X_test,y_test=deep_utils.load_pickle_files(X_test_files, y_test_files)
    
    #Shuffle, reshape, and normalize testing data
    X_test,y_test=deep_utils.simul_shuffle(X_test,y_test) 
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]).astype(np.uint8)
    y_test = y_test.reshape((y_test.shape[0],1,-1)).astype(np.uint8)
    y_test = y_test.squeeze()
    X_test=np.divide(X_test,255).astype(np.float16)
    y_test=np.divide(y_test,255).astype(np.float16)    
    
    model=model()
    model.compile(loss='mean_squared_error',optimizer=Adam(),metrics=['mse'])      

    #Save best model weights checkpoint
    filepath="weights_latest.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    #Tensorboard setup
    log_dir = r"logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")        
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    
    callbacks_list = [checkpoint, tensorboard_callback]
    
    model.fit_generator(_batchGenerator(X_files,y_files,batch_size),
                        epochs=num_epochs,
                        steps_per_epoch=X_test.shape[0]//batch_size,
                        max_queue_size=1,
                        validation_data=(X_test,y_test),
                        callbacks=callbacks_list,
                        verbose=2)
    
    return model
    
if __name__=='__main__':
    training_models=[models.cnn, 
                     models.pretrained_unet_cnn,
                     models.rcnn_640_480, 
                     models.pretrained_unet_rcnn,
                     models.pretrained_unet, 
                     models.wnet, 
                     models.wnet_connected, 
                     ]
    #Specify model argument to main()
    model=main(model=training_models[4],num_epochs=40,batch_size=2)