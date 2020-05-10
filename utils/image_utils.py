# -*- coding: utf-8 -*-
"""
Image Utility functions.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

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

def image_from_np(image_array,save=False,rgb=True):
    '''Plots RGB or grayscale image from numpy array'''
    if rgb==True:
        img = Image.fromarray(image_array, 'RGB')
        img.show()
    else:
        #img = Image.fromarray(image_array, '1')
        img=cv2.imshow('image_from_np',image_array)
        cv2.waitKey(0)
    return img

def add_blur(im_array,ksize=12,sigmaColor=400,sigmaMax=700):
    """
    Adds bilateral filtering to blur objects but preserve edges
    """
    return cv2.bilateralFiltering(im_array,ksize,sigmaColor,sigmaMax)

def rgb_read(filename):
    '''Reads RGB image from png file and returns it as a numpy array'''
    #Load image
    image=Image.open(filename)
    #store as np.array
    rgb=np.array(image)
    image.close()
    return rgb

def depth_read(filename):
    '''Loads depth map D from png file and returns it as a numpy array'''
    #Lower is closer
    # From KITTI devkit
    
    image=Image.open(filename)
    depth_png = np.array(image, dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!

    if depth_png.shape==(480,640,3):
        depth_png=(depth_png[:,:,0]+depth_png[:,:,1]+depth_png[:,:,2])/3
    
    #depth_png=depth_png[:,:,3]
    assert(np.max(depth_png) <= 255)
    depth=depth_png.astype(np.float)
    #depth = depth_png.astype(np.float) / 256.
    #depth[depth_png == 0] = -1.
    image.close()

    return depth

def depth_read_kitti(filename):
    '''Loads depth map D from png file and returns it as a numpy array'''
    #Lower is closer
    # From KITTI devkit
    
    image=Image.open(filename)
    depth_png = np.array(image, dtype=int)
    
    #TODO: Determine if this if legitimate for getting depth values
    if depth_png.shape==(192,640,4):
        # print('it is')
        depth_png=(depth_png[:,:,0]+depth_png[:,:,1]+depth_png[:,:,2])/3
    
    assert(np.max(depth_png) <= 255)
    depth=depth_png.astype('int8') #np.float
 
    image.close()

    return depth
def heatmap(image,save=False,name='heatmap',cmap='gray'):
    '''Plots heatmap of depth data from image or np.ndarray.'''
    if type(image)==np.ndarray:
        pic_array=image
    else:
        #Convert to np.ndarray
        pic=Image.open(image)
        pic_array=np.array(pic)
    #Plot heatmap
    print(pic_array.shape)
    plt.imshow(pic_array, cmap=cmap, interpolation='nearest') #cmap=binary, plasma, gray
    plt.show()
    if save==True:
        plt.imsave(name+'.png',pic_array, cmap=cmap)
    
if __name__=='__main__':
    filename=r"G:\Documents\KITTI\sandbox\y_depth\2011_09_26_drive_0002_sync\proj_depth\groundtruth\image_02\0000000005.png"
    heatmap(filename)
    d=depth_read(filename)
