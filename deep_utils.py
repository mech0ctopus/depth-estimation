# -*- coding: utf-8 -*-
"""
Utility functions for working with neural networks.
"""
from keras.models import model_from_json, model_from_yaml
import numpy as np
from glob import glob
import pickle
import matplotlib.pyplot as plt
from image_utils import depth_read, rgb_read
import random

def save_model(model,serialize_type,model_name='model'):
    '''Saves model and weights to file.'''
    serialize_type=serialize_type.lower()
    
    if serialize_type=='yaml':
        model_yaml = model.to_yaml()
        with open(model_name+".yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
    elif serialize_type=='json':
        model_json = model.to_json()
        with open(model_name+".json", "w") as json_file:
            json_file.write(model_json)
    model.save_weights(model_name+".h5")
    
    print(model_name+' & weights saved to disk.')

def load_model(model_filepath,weights_filepath):
    '''Loads model and weights to file.'''
    
    serialize_type=model_filepath.split('.')[-1]
    serialize_type=serialize_type.lower()
    
    file = open(model_filepath, 'r')
    loaded_model = file.read()
    file.close()
    
    if serialize_type=='yaml':
        loaded_model = model_from_yaml(loaded_model)
    elif serialize_type=='json':
        loaded_model=model_from_json(loaded_model)

    loaded_model.load_weights(weights_filepath)
    print("Loaded model from disk")
    return loaded_model

def get_layer_names(model):
    '''Returns list of layer names.'''
    layer_names=[]
    for layer in model.layers:
        layer_names.append(layer.name)
    return layer_names

def generate_pickle_files(X,y):
    '''Generates pickle file to compress whole dataset.'''
    pickle.dump(X, open(r"X.p", "wb"), protocol=4)
    pickle.dump(y, open(r"y.p", "wb"), protocol=4)

def load_pickle_files(X_file, y_file):
    '''Reads data from pickle files'''
    X=pickle.load(open(X_file,'rb'))
    y=pickle.load(open(y_file,'rb'))
    return X, y

def read_data(data_folderpath,num_images):
    '''Reads full dataset.  Assumes data has been resized.
    Assumes "data_folderpath" contains subfolders corresponding
    to class names and each containing jpg files for class.'''
    
    X=np.zeros((num_images,375,1242,3),dtype=np.uint8) #Full set: 67 for image02 and 67 for image03
    y=np.zeros((num_images,375,1242),dtype=np.uint8)
    
    #Append folderpaths if needed
    if data_folderpath.endswith('\\')==False:
        data_folderpath=str(data_folderpath)+ '\\'
    #Establish base paths
    x_basepath=data_folderpath+'X_rgb'+'\\'
    y_basepath=data_folderpath+'y_depth'+'\\'
    #Collect all foldernames for y samples
    y_foldernames=glob(y_basepath+'*/')
 
    #Build full list of left and right depth images
    left_depth_images=[]
    right_depth_images=[]
    y_image_names={}
    count=0
    for foldername in y_foldernames:
        left_depth=foldername+'proj_depth\groundtruth\image_02\\'
        right_depth=foldername+'proj_depth\groundtruth\image_03\\' 
        left_depth_paths=glob(left_depth+'*.PNG')           
        right_depth_paths=glob(right_depth+'*.PNG')
        [left_depth_images.append(path) for path in left_depth_paths]
        [right_depth_images.append(path) for path in right_depth_paths]
        #Build list of image names we have depth data for
        y_image_names[count]=[filepath.split('\\')[-1] for filepath in left_depth_paths]
        count+=1

    #Read Depth images
    count=0
    for depth_images in [left_depth_images,right_depth_images]:
        #Loop through all depth images
        for image in depth_images:
            y[count]=depth_read(image)
            count+=1

    #Collect all foldernames for x samples
    x_foldernames=glob(x_basepath+'*/')

    #Build full list of left and right RGB images
    left_rgb_images=[]
    right_rgb_images=[]
    count=0
    for foldername in x_foldernames:
        left_rgb=foldername+'image_02\data\\'
        right_rgb=foldername+'image_03\data\\'
        #Build list of paths
        for image_name in y_image_names[count]:
            left_rgb_images.append(left_rgb+image_name)
            right_rgb_images.append(right_rgb+image_name)
        count+=1
        
    #Read RGB images
    count=0
    for rgb_images in [left_rgb_images,right_rgb_images]:
        #Loop through all rgb images
        for image in rgb_images:
            X[count]=rgb_read(image)
            count+=1
                
    print('Pickling')
    generate_pickle_files(X,y)
    return X,y 

def simul_shuffle(mat1, mat2):
    '''Shuffles two matrices in the same order'''
    idx=np.arange(0,mat1.shape[0])
    random.shuffle(idx)
    mat1=mat1[idx]
    mat2=mat2[idx]
    return mat1, mat2
    
def plot_accuracy(history):
    '''Summarize history for accuracy'''
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_mse(history):
    '''Summarize history for mean-squared error (MSE)'''
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model MSE')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def plot_loss(history):
    '''Summarize history for loss'''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
if __name__ == '__main__':   
    dataset=r"G:\Documents\KITTI\sandbox"
    num_images=898
    X,y=read_data(dataset,num_images)