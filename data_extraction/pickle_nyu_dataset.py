# -*- coding: utf-8 -*-
"""
Pickles thin NYU Depth dataset
"""
import numpy as np
from glob import glob
from PIL import Image
import pickle
from utils.image_utils import depth_read

def generate_pickle_files(X,y):
    '''Generates pickle file to compress whole dataset.'''
    pickle.dump(X, open(r"X.p", "wb"), protocol=4)
    pickle.dump(y, open(r"y.p", "wb"), protocol=4)

def load_pickle_files(X_file, y_file):
    '''Reads data from pickle files'''
    X=pickle.load(open(X_file,'rb'))
    y=pickle.load(open(y_file,'rb'))
    return X, y

def read_data(data_folderpath,output_folderpath,num_intervals=35):
    '''Reads full dataset.  Assumes data has been resized.
    Assumes "data_folderpath" contains subfolders corresponding
    to class names and each containing jpg files for class.'''
    print('Initializing Matrices')
    X=np.zeros((7392,480,640,3),dtype=np.uint8) #Was 480,640
    y=np.zeros((7392,480,640),dtype=np.uint8)   #Was 480,640

    #Append folderpaths if needed
    if data_folderpath.endswith('\\')==False:
        data_folderpath=str(data_folderpath)+ '\\'
    if output_folderpath.endswith('\\')==False:
        output_folderpath=str(output_folderpath)+ '\\'
    X_folderpath=data_folderpath+'X_rgb\\'
    y_folderpath=data_folderpath+'y_depth\\'
    
    #Build list of filenames
    X_filelist=glob(X_folderpath+'*.png')
    y_filelist=glob(y_folderpath+'*.png')
    
    X_filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    y_filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    for idx in range(len(X_filelist)):
        print(f'Reading file #{idx}')
        #Load images
        rgb_image=Image.open(X_filelist[idx])
        #depth_image=Image.open(y_filelist[idx])

        #store as np.arrays
        X[idx]=np.array(rgb_image) #.reshape(640,480,3) #Reshape is new
        #y[idx]=np.array(depth_image) #.reshape(640,480) #Reshape is new
        y[idx]=depth_read(y_filelist[idx])
#        if idx==1:
##            image_utils.image_from_np(y[idx],rgb=False)
#            image_utils.image_from_np(X[idx])
#            plt.imshow(y[idx], cmap='gray', interpolation='nearest')
#            break
        rgb_image.close()
        #depth_image.close()

    print('Splitting Data')
    y_splits=np.array_split(y,num_intervals)
    X_splits=np.array_split(X,num_intervals)
    X=None
    y=None
    
    print('Pickling')
    for idx in range(len(y_splits)):
        #Save to pickle file
        pickle.dump(y_splits[idx], open(output_folderpath+f"y_{idx}.p", "wb"), protocol=4)
        pickle.dump(X_splits[idx], open(output_folderpath+f"X_{idx}.p", "wb"), protocol=4)
    
if __name__ == '__main__':   
    dataset=r"E:\NYU\nyud_raw_data\nyuv2-python-toolbox-master\colorized"
    output_folderpath=r"G:\Documents\NYU Depth Dataset\nyu_data\pickled_colorized"
    read_data(dataset,output_folderpath)