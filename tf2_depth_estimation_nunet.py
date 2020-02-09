# Kyle J. Cantrell & Craig D. Miller
# kjcantrell@wpi.edu & cdmiller@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Depth Estimation from RGB Images

import numpy as np
from glob import glob
from utils import deep_utils
from utils.image_utils import depth_read, rgb_read
from models import models
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import datetime
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import segmentation_models

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def _batchGenerator(X_filelist,y_filelist,batchSize):
    """
    Yield X and Y data when the batch is filled.
    """
    #Sort filelists to confirm they are same order
    X_filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    y_filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    #Shuffle order of filenames
    X_filelist,y_filelist=deep_utils.simul_shuffle(X_filelist,y_filelist)

    idx=0
    while True:
        while idx<len(X_filelist):
            X_train=np.zeros((batchSize,480,640,3),dtype=np.uint8)
            y_train=np.zeros((batchSize,480,640),dtype=np.uint8)
            
            for i in range(batchSize):
                #Load images
                X_train[i]=rgb_read(X_filelist[idx+i])
                y_train[i]=depth_read(y_filelist[idx+i])
    
            #Reshape [samples][width][height][pixels]
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 
                                      X_train.shape[2], X_train.shape[3]).astype(np.uint8)

            y_train = y_train.reshape((y_train.shape[0],1,-1)).astype(np.uint8)
            y_train = y_train.squeeze()
                 
            # normalize inputs and outputs from 0-255 to 0-1
            X_train=np.divide(X_train,255).astype(np.float16)   
            y_train=np.divide(y_train,255).astype(np.float16)
            
            if (idx % 1024)==0:
                print(str(idx)+'/'+str(len(X_filelist)))
                
            idx+=batchSize
            
            yield X_train, y_train
            
def main(model_name, model=models.wnet_connected,num_epochs=5,batch_size=2):
    '''Trains depth estimation model.'''
    
    segmentation_models.set_framework('tf.keras')
    print(segmentation_models.framework())
    
    #Build list of filenames
    X_folderpath=r"E:\NYU\nyud_raw_data\nyuv2-python-toolbox-master\colorized\X_rgb\\"
    y_folderpath=r"E:\NYU\nyud_raw_data\nyuv2-python-toolbox-master\colorized\y_depth\\"
    X_filelist=glob(X_folderpath+'*.png')
    y_filelist=glob(y_folderpath+'*.png')
    
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
    filepath=f"{model_name}_weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min')
    
    #Tensorboard setup
    log_dir = f"logs\\{model_name}\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")        
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    
    callbacks_list = [checkpoint, tensorboard_callback]
    
    model.fit_generator(_batchGenerator(X_filelist,y_filelist,batch_size),
                        epochs=num_epochs,
                        steps_per_epoch=len(X_filelist)//batch_size,
                        validation_data=(X_test,y_test),
                        max_queue_size=1,
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
                     models.wnet_connected]
    model_names=['CNN',
                 'U-Net_CNN',
                 'RCNN',
                 'U-Net_RCNN',
                 'U-Net',
                 'W-Net',
                 'W-Net_Connected']
    
    #Specify test_id argument to main()
    test_id=4
    model=main(model_name=model_names[test_id],model=training_models[test_id],
               num_epochs=10,batch_size=2)
    
    #Save model
    deep_utils.save_model(model,serialize_type='yaml',
                          model_name=f'{model_names[test_id]}_nyu_model',
                          save_weights=False)