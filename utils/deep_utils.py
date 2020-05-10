# -*- coding: utf-8 -*-
"""
Utility functions for working with neural networks.
"""
from keras.models import model_from_json, model_from_yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from tensorflow.keras.optimizers import Adam

def save_model(model,serialize_type,model_name='model',save_weights=False):
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
    if save_weights:
        model.save_weights(model_name+".h5")
        print(model_name+' & weights saved to disk.')
    else:
        print(model_name+' saved to disk.')

    
def load_model_weights(model, weights):
    '''Loads pretrained and compiled model.'''
    model.compile(loss='mean_squared_error',optimizer=Adam(),metrics=['mse']) 
    model.load_weights(weights)
    return model

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

def simul_shuffle(mat1, mat2):
    '''Shuffles two matrices in the same order'''
    
    if type(mat1)==list:
        temp = list(zip(mat1, mat2)) 
        random.shuffle(temp) 
        mat1, mat2 = zip(*temp)
    else:
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

def plot_full_val_loss(history):
    '''Summarize history for loss'''
    loss_history=[]
    val_loss_history=[]
    for h in history:
        for item in h.history['loss']:
            loss_history.append(item)
        for item in h.history['val_loss']:
            val_loss_history.append(item)
    plt.plot(loss_history)
    plt.plot(val_loss_history)
    plt.title('Validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':   
    dataset=r"G:\Documents\KITTI\sandbox"