# Kyle J. Cantrell & Craig D. Miller
# kjcantrell@wpi.edu & cmiller@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Depth Estimation from RGB Images

import numpy as np
import tensorflow as tf
from keras.utils import plot_model
import time
from glob import glob
import matplotlib.pyplot as plt
from segmentation_models import Unet
from utils import deep_utils
from utils import image_utils
from models import models
from models.losses import sil

start=time.time()

#Initialize tensorflow GPU settings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#Load training data
pickle_files_folderpath=r"G:\Documents\NYU Depth Dataset\nyu_data\pickled_new"
X_files=glob(pickle_files_folderpath+'\\X_*')
y_files=glob(pickle_files_folderpath+'\\y_*')

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

num_training_batches=len(X_files)
history=[]

def main(model=models.wnet_connected,num_epochs=5,batch_size=2):
    '''Trains depth estimation model.'''
    #Loop through all training batches
    for i in range(num_training_batches):  
        print('Batch '+str(i)+': '+'Loading data')
        X_train,y_train=deep_utils.load_pickle_files(X_files[i], y_files[i])
        X_train,y_train=deep_utils.simul_shuffle(X_train,y_train)
        
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
            model=model()
    
        print('Batch '+str(i)+': '+'Fitting model')
        history.append(model.fit(X_train, y_train,validation_data=(X_test, y_test), 
                                 epochs=num_epochs, batch_size=batch_size, verbose=2,))
        
        plt.figure()
        deep_utils.plot_loss(history[i])
        
        #Show test image
        if i==0:
            image_utils.image_from_np(np.multiply(X_test[0],255).astype(np.uint8))
            
        #Evaluate test image and print depth prediction
        test_image=X_test[0].reshape(1,X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2])
        y_est=model.predict(test_image)
        y_est=y_est.reshape((X_train.shape[1],X_train.shape[2]))*255 #De-normalize for depth viewing
        print('Sample image, Batch '+str(i))
        image_utils.heatmap(y_est,save=True,name='Batch '+str(i))
    
    print(model.summary())
    finish=time.time()
    elapsed=finish-start
    print('Runtime :'+str(elapsed)+' seconds')
    
    deep_utils.plot_full_val_loss(history)
    
    #Save model, weights, and architecture
    deep_utils.save_model(model,serialize_type='yaml',model_name='depth_estimation_unet_nyu_model')
    plot_model(model, to_file='model.png')
        
    #Show a few test image and predicted depth results
    for i in [0,1,2]:  
        image_utils.image_from_np(np.multiply(X_test[i],255).astype(np.uint8))  #De-normalize for viewing
        test_image=X_test[i].reshape(1,X_test[i].shape[0],X_test[i].shape[1],X_test[i].shape[2])
        y_est=model.predict(test_image)
        y_est=y_est.reshape((X_train.shape[1],X_train.shape[2]))*255 #De-normalize for depth viewing
        print('Sample image, X_test['+str(i)+']:')
        image_utils.heatmap(y_est)
    
    #Test a new image outside of training & testing dataset
    test_image=image_utils.rgb_read(r"C:\Users\Craig\Desktop\test5.png") #640x480
    test_image=test_image.reshape(1,X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2])
    test_image=np.divide(test_image,255).astype(np.float16) #Normalize for prediction
    y_est=model.predict(test_image)
    y_est=y_est.reshape((X_train.shape[1],X_train.shape[2]))*255 #De-normalize for depth viewing
    print('New Test Image:')
    image_utils.heatmap(y_est)
    
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
    main(model=training_models[6])