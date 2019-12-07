# -*- coding: utf-8 -*-
"""
Reads and pickles KITTI dataset into multiple files.

Expects dataset structured as follows:
    dataset
        X_rgb
            2011_09_26_drive_0002_sync (default KITTI subtree)
            2011_09_26_drive_0009_sync
            ...
        y_depth
            2011_09_26_drive_0002_sync
            2011_09_26_drive_0009_sync
            ...

Usage:
pickle_data(dataset,output_folderpath)

"""
import numpy as np
from glob import glob
from utils.image_utils import depth_read, rgb_read
import pickle
from math import ceil

def append_folderpath(folderpath):
    '''Adds '\\' to folderpath end if needed'''
    if folderpath.endswith('\\')==False:
        folderpath=str(folderpath)+ '\\'
        
    return folderpath
        
def get_X_y_paths(data_folderpath,subfolder):
    '''Builds X & y subfolderpaths.'''
    #Append folder paths if necessary
    data_folderpath=append_folderpath(data_folderpath)
    subfolder=append_folderpath(subfolder)
    #Establish X and y subfolder paths
    X_rgb_subpath=data_folderpath+'X_rgb'+'\\'+subfolder
    y_depth_subpath=data_folderpath+'y_depth'+'\\'+subfolder
    
    return X_rgb_subpath, y_depth_subpath

def get_depth_paths(y_depth_subpath):
    '''Build lists of filepaths for left & right image depths .'''
    y_depth_subpath=append_folderpath(y_depth_subpath)
    #Point to correct location
    left_depth=y_depth_subpath+'proj_depth\groundtruth\image_02\\'
    right_depth=y_depth_subpath+'proj_depth\groundtruth\image_03\\' 
    #Get all image filenames
    left_depth_paths=glob(left_depth+'*.PNG')           
    right_depth_paths=glob(right_depth+'*.PNG')
    
    return left_depth_paths, right_depth_paths
    
def pickle_depth_images(subfolder,depth_paths,output_folderpath,max_array_len=200):
    '''Generates pickle file of y_depth data'''
    #Read depth images and update np.array
    num_images=len(depth_paths)
    y=np.zeros((num_images,375,1242),dtype=np.uint8)
    for idx, depth_path in enumerate(depth_paths):
        y[idx]=depth_read(depth_path)
    
    #Split data into smaller pickle files if necessary
    num_intervals=ceil(num_images/max_array_len)
    if num_intervals>1:
        y_splits=np.array_split(y,num_intervals)
        #Clear y variable
        y=None
        for idx, y_split in enumerate(y_splits):
            #Save to pickle file
            pickle.dump(y_split, open(output_folderpath+r"y_"+str(subfolder)+f"_{idx}.p", "wb"), protocol=4)
            #Clear y_split variable
        y_splits=None
    else:
        #Save to pickle file
        pickle.dump(y, open(output_folderpath+r"y_"+str(subfolder)+".p", "wb"), protocol=4)
        #Clear y variable
        y=None
    
def get_rgb_paths(X_rgb_subpath,left_depth_paths, right_depth_paths):
    '''Create list of RGB paths corresponding to input depth paths'''
    #Point to correct location
    left_rgb=X_rgb_subpath+'image_02\data\\'
    right_rgb=X_rgb_subpath+'image_03\data\\'
    #Build list of image names in left and right depth paths
    left_depth_image_names=[filepath.split('\\')[-1] for filepath in left_depth_paths]
    right_depth_image_names=[filepath.split('\\')[-1] for filepath in right_depth_paths]
    #Build list of left and right RGB paths corrseponding to depth images
    left_rgb_paths,right_rgb_paths=[],[]
    for left_depth_image_name in left_depth_image_names:
        left_rgb_paths.append(left_rgb+left_depth_image_name)
    for right_depth_image_name in right_depth_image_names:
        right_rgb_paths.append(right_rgb+right_depth_image_name)    
    rgb_paths=left_rgb_paths+right_rgb_paths
    
    return rgb_paths
 
def pickle_rgb_images(subfolder,rgb_paths,output_folderpath,max_array_len=200):
    '''Generates pickle file of X_rgb data'''
    #Read RGB images and update np.array
    num_images=len(rgb_paths)
    X=np.zeros((num_images,375,1242,3),dtype=np.uint8)
    for idx, rgb_path in enumerate(rgb_paths):
        X[idx]=rgb_read(rgb_path)
        
    #Split data into smaller pickle files if necessary
    num_intervals=ceil(num_images/max_array_len)
    if num_intervals>1:
        X_splits=np.array_split(X,num_intervals)
        #Clear X variable
        X=None
        for idx, X_split in enumerate(X_splits):
            #Save to pickle file
            pickle.dump(X_split, open(output_folderpath+r"X_"+str(subfolder)+f"_{idx}.p", "wb"), protocol=4)
            #Clear X_split variable
        X_splits=None
    else:
        #Save to pickle file
        pickle.dump(X, open(output_folderpath+r"X_"+str(subfolder)+".p", "wb"), protocol=4)
        #Clear X variable
        X=None
        
def pickle_folder(data_folderpath,subfolder,output_folderpath):
    '''Reads and pickles one folder from KITTI.  
    Save X, y folder pair as pickle files.'''
    #Identify where X and y data is located    
    X_rgb_subpath, y_depth_subpath=get_X_y_paths(data_folderpath,subfolder)
    #Build list of filepaths for left & right image depths
    left_depth_paths, right_depth_paths=get_depth_paths(y_depth_subpath)
    depth_paths=left_depth_paths+right_depth_paths
    #Create pickle file of all depth images listed in depth_paths
    pickle_depth_images(subfolder,depth_paths,output_folderpath)
    #Create list of corresponding RGB paths
    rgb_paths=get_rgb_paths(X_rgb_subpath,left_depth_paths, right_depth_paths)
    #Create pickle file of all RGB images listed in rgb_paths
    pickle_rgb_images(subfolder,rgb_paths,output_folderpath)

def pickle_dataset(data_folderpath,output_folderpath):
    '''Reads and pickles KITTI dataset into multiple files.'''
    output_folderpath=append_folderpath(output_folderpath)
    #Build list of subfolders in data_folderpath\y_depth
    data_folderpath=append_folderpath(data_folderpath)
    y_depth_path=data_folderpath+'y_depth'
    subfolders=glob(y_depth_path+'\\*\\')
    #Parse out foldername
    subfolders=[subfolder.split('\\')[-2] for subfolder in subfolders]
    #Pickle each subfolder
    for subfolder in subfolders:
        print(f'Pickling {subfolder}')
        pickle_folder(data_folderpath,subfolder,output_folderpath)
    
if __name__ == '__main__':   
    dataset=r"G:\Documents\KITTI\sandbox_val"
    output_folderpath=r"G:\Documents\KITTI\pickled_KITTI\validation"
    pickle_dataset(dataset,output_folderpath)