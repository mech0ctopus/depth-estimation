# -*- coding: utf-8 -*-
"""
Image Utility functions.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


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
    # From KITTI devkit
    image=Image.open(filename)
    depth_png = np.array(image, dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) <= 255)

    depth = depth_png.astype(np.float) / 256.
    #depth[depth_png == 0] = -1.
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
    plt.imshow(pic_array, cmap=cmap, interpolation='nearest') #cmap=binary, plasma, gray
    plt.show()
    if save==True:
        plt.imsave(name+'.png',pic_array, cmap=cmap)
    
if __name__=='__main__':
    filename=r"G:\Documents\KITTI\sandbox\y_depth\2011_09_26_drive_0002_sync\proj_depth\groundtruth\image_02\0000000005.png"
    heatmap(filename)
    d=depth_read(filename)
