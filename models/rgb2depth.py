# -*- coding: utf-8 -*-
"""
Tool for creating a depth prediction video from an input RGB video.
"""
import cv2
import numpy as np
import models
from tensorflow.keras.optimizers import Adam
  
def rgb2depth_video(filename, model, out_FPS=10, width=640, height=192,
                    output_filename=r'depth_output.avi', gamma=1,
                    mirror=False, crop_mode='middle'):
    '''Create depth prediction video from input RGB video. Adjust 
    brightness (gamma) for viewing purposes only. Anything other than 
    gamma=1 is distorting the numerical depth prediction.'''
        
    cam = cv2.VideoCapture(filename)

    # Define the codec and create VideoWriter object.
    out = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc('M','J','P','G'), 
                          out_FPS, (int(width),int(height)))
      
    while(cam.isOpened()):
        ret_val, img = cam.read()
        
        try:
            if mirror: 
                img = cv2.flip(img, 1)

            #If img doesn't match output height & width
            if (img.shape[0] != height) or (img.shape[1] != width):
                #Crop image
                img=crop_image(img,width,height,mode=crop_mode)
    
            #Predict depth
            img=img.reshape(1,height,width,3)
            img=np.divide(img,255).astype(np.float16) #Normalize input
            y_est=model.predict(img)
            y_est=y_est.reshape((height,width))
            
            #Map 2D grayscale to RGB equivalent
            vis = cv2.cvtColor((y_est*255*(1/gamma)).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            vis = cv2.cvtColor(vis,cv2.COLOR_BGR2GRAY)
            #Map BGR to Rainbow
            vis=cv2.applyColorMap(vis,cv2.COLORMAP_RAINBOW)
            
            #Write prediction to video
            out.write(vis)
        except:
            break
        
    cam.release()
    cv2.destroyAllWindows()

def crop_image(img,x=640,y=192,mode='middle'):
    '''Crops images starting at 'top', 'middle', or 'bottom'.'''
   
    if img.shape[0] != y:
        #Crop vertically
        if mode=='top':
            y_top, y_bottom = 0, y
            img=img[y_top:y_bottom,:]
        elif mode=='middle':
            y_mid=img.shape[0]/2
            y_top, y_bottom = int(y_mid-y/2), int(y_mid+y/2)
            img=img[y_top:y_bottom,:]
        elif mode=='bottom':
            y_top, y_bottom = img.shape[0]-y, img.shape[0]
            img=img[y_top:y_bottom,:]
        else:
            print('Unknown crop mode.')
            img=None
    
    if img.shape[1] != x:
        #Crop horizontally in the middle of image
        x_mid=img.shape[1]/2
        x_left, x_right = int(x_mid-x/2), int(x_mid+x/2)
        img=img[:,x_left:x_right]
        
    return img

def load_model(model, weights):
    '''Loads pretrained and compiled model.'''
    model.compile(loss='mean_squared_error',optimizer=Adam(),metrics=['mse']) 
    model.load_weights(weights)
    return model

if __name__=='__main__':
    #Load pretrained & compiled model
    weights=r"E:\W-Net_Connected_weights_best_KITTI_35Epochs.hdf5"
    model=models.wnet_connected()
    model=load_model(model,weights)
    #Define input RGB video (tested with .mp4, .mov, and .avi)
    rgb_video=r"G:\Program Files\MATLAB\R2018b\toolbox\vision\visiondata\atrium.mp4"
    #Create depth video
    rgb2depth_video(rgb_video,model,out_FPS=30, width=640, height=192,
                    output_filename=r'depth_output.avi', gamma=0.6,
                    mirror=False, crop_mode='middle')