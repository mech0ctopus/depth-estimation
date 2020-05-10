# -*- coding: utf-8 -*-
"""
Image data generator

https://stackoverflow.com/questions/47826730/how-to-save-resized-images-using-imagedatagenerator-and-flow-from-directory-in-k
"""
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
from os.path import basename
import cv2

datagen = ImageDataGenerator(featurewise_center=False, 
                             samplewise_center=False, 
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False, 
                             zca_epsilon=1e-06, 
                             rotation_range=0, 
                             width_shift_range=0.0, 
                             height_shift_range=0.0, 
                             brightness_range=(0.3,1), #Use
                             shear_range=0.0, 
                             zoom_range=0.0, 
                             channel_shift_range=0.0, 
                             fill_mode='nearest', 
                             cval=0.0, 
                             horizontal_flip=False, 
                             vertical_flip=False, 
                             rescale=None, 
                             preprocessing_function=None, 
                             data_format='channels_last', 
                             validation_split=0.0, 
                             interpolation_order=1)

folderpath=r"C:\Users\Craig\Desktop\test_rgb_folder\\"
filelist=glob(folderpath+'*.png')
save_here = r'C:\Users\Craig\Desktop\test_rgb_folder\augmented'

for filepath in filelist:
    image = np.expand_dims(cv2.imread(filepath), 0)
    datagen.fit(image)
    
    idx=basename(filepath).split('.')[0]
    idx=idx.split('_')[-1]
    
    depth_file=r"C:\Users\Craig\Desktop\test_depth_folder\\"+f"d_{idx}.PNG"
    im=cv2.imread(depth_file)
    
    cv2.imwrite(r"C:\Users\Craig\Desktop\test_depth_folder\augmented\\"+f"z_aug_d_{idx}_000000.PNG",im)
    
    for x, val in zip(datagen.flow(image,                    #image we chose
            save_to_dir=save_here,     #this is where we figure out where to save
             save_prefix=f'z_aug_rgb_{idx}',        # it will save the images as 'aug_0912' some number for every new augmented image
            save_format='png'),range(10)) :     # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
        cv2.imwrite(r"C:\Users\Craig\Desktop\test_depth_folder\augmented\\"+f"z_aug_d_{idx}_{val}.PNG",im)
        
