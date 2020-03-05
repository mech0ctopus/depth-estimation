# -*- coding: utf-8 -*-
"""
Tool for testing depth estimation models on live video stream.
"""
import cv2
import numpy as np
import deep_utils
import image_utils
import time
from models import models
from tensorflow.keras.optimizers import Adam
  
def video_stream(model,method='cv2',mirror=True):
    '''Runs depth estimation on live webcam video stream'''
    #Load model
    #model=deep_utils.load_model(model,weights)
    cam = cv2.VideoCapture(0)
    
    while True:
        start=time.time()
        
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
            
        #Resize image
        img=cv2.resize(img,(int(640),int(480)))
        img=img.reshape(1,480,640,3)
        img=np.divide(img,255).astype(np.float16)
        #Predict depth
        y_est=model.predict(img)
        y_est=y_est.reshape((480,640)) #cv2 maps 0-1 to 0-255

        #Show depth prediction results
        if method=='cv2':
            #Map 2D grayscale to RGB equivalent
            vis = cv2.cvtColor(y_est.astype(np.float32), cv2.COLOR_GRAY2BGR)
            cv2.imshow('Depth Estimate', vis)
        elif method=='heatmap':
            image_utils.heatmap(y_est,cmap='plasma')
        else:
            print('Unknown display method.')
        
        #Estimate instantaneous frames per second
        end=time.time()
        fps=round(1/(end-start),2)        
        print(f'FPS: {fps}')

        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    #Load weights (h5)
    weights={'unet':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Weights & Models\20Epochs_NoAugment_LowerLR\20200213-065621\U-Net_weights_best.hdf5",
            'wnet':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Weights & Models\20Epochs_NoAugment_LowerLR\20200213-082137\W-Net_weights_best.hdf5",
            'wnet_c':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Weights & Models\20Epochs_NoAugment_LowerLR\20200212-205108\W-Net_Connected_weights_best.hdf5"
             }
    
    display_methods=['cv2','heatmap']
    
    model_name='wnet_c'
    model=models.wnet_connected()

    model.compile(loss='mean_squared_error',optimizer=Adam(),metrics=['mse']) 
    model.load_weights(weights['wnet_c'])

    video_stream(model,method=display_methods[0],mirror=True)