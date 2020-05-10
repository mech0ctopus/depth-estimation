# -*- coding: utf-8 -*-
"""
Tool for testing depth estimation models on live video stream.
"""
import cv2
import numpy as np
import image_utils
import deep_utils
import time
from models import models
  
def rgb2depth_stream(model,method='cv2',mirror=True,width=640,height=192,
                 gamma=1,crop_mode='middle'):
    '''Runs depth estimation on live webcam video stream.  Adjust 
    brightness (gamma) for viewing purposes only. Anything other than 
    gamma=1 is distorting the numerical depth prediction.'''
    cam = cv2.VideoCapture(0)
    
    while True:
        start=time.time()
        
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
            
        #If img doesn't match output height & width
        if (img.shape[0] != height) or (img.shape[1] != width):
            #Crop image
            img=image_utils.crop_image(img,width,height,mode=crop_mode)
  
        img=img.reshape(1,height,width,3)
        img=np.divide(img,255).astype(np.float16)
        #Predict depth
        y_est=model.predict(img)
        y_est=y_est.reshape((height,width))

        #Show depth prediction results
        if method=='cv2':        
            #Map 2D grayscale to RGB equivalent
            vis = cv2.cvtColor((y_est*255*(1/gamma)).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            vis = cv2.cvtColor(vis,cv2.COLOR_BGR2GRAY)
            #Map BGR to Rainbow
            vis=cv2.applyColorMap(vis,cv2.COLORMAP_RAINBOW)
            
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
    #Load pretrained & compiled model
    weights=r"E:\W-Net_Connected_weights_best_KITTI_35Epochs.hdf5"
    model=models.wnet_connected()
    model=deep_utils.load_model_weights(model,weights)
    
    display_methods=['cv2','heatmap']

    rgb2depth_stream(model,method=display_methods[0],mirror=True,
                     width=640,height=192,gamma=0.3,crop_mode='middle')