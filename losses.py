# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:46:34 2019
LOSS Functions 
@author: kjcantrell99
"""
import keras.backend as k
n = 1242*375

def sil(yTrue,yPred):
    """
    Scale-Invariant Loss
    """
    if not K.is_tensor(yPred):
        yPred = K.constant(yPred)
    yTrue = K.cast(yTrue, yPred.dtype)
    first_log = K.log(K.clip(yPred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(yTrue, K.epsilon(), None) + 1.)
    di = first_log - second_log
    term1 = K.sum(K.square(di),axis=-1)
    term2 = K.square(K.sum(di),axis=-1)
    return (1/n)*(term1) - (0.5/(n*n))*term2



