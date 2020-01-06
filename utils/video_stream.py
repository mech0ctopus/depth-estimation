# -*- coding: utf-8 -*-
"""
Tool for testing depth estimation models on live video stream.
"""
import cv2
import numpy as np
import deep_utils
import image_utils
import time
    
def video_stream(model,weights,method='cv2',mirror=True):
    '''Runs depth estimation on live webcam video stream'''
    #Load model
    model=deep_utils.load_model(model,weights)
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
    display_methods=['cv2','heatmap']
    test_model='wnetc'
    
    #Test using pretrained models & weights
    if test_model=='wnetc':
        model=r"Weights & Models\200 Epochs\W-Net Connected\MSE\1e-3\depth_estimation_wnetc_mse_nyu_model.yaml"
        weights=r"Weights & Models\200 Epochs\W-Net Connected\MSE\1e-3\depth_estimation_wnetc_mse_nyu_model.h5"
    elif test_model=='unet':
        model=r"Weights & Models\200 Epochs\U-Net\MSE\1e-3\depth_estimation_unet_mse_nyu_model.yaml"
        weights=r"Weights & Models\200 Epochs\U-Net\MSE\1e-3\depth_estimation_unet_mse_nyu_model.h5"

    video_stream(model,weights,method=display_methods[0],mirror=True)