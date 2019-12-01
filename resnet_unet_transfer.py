# -*- coding: utf-8 -*-
"""
ResNet U-Net Segmentation Transfer Learning
"""

from keras_segmentation.pretrained import resnet_pspnet_VOC12_v0_1, pspnet_50_ADE_20K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.optimizers import Adam
import tensorflow as tf

premodel = pspnet_50_ADE_20K()

#pretrained_model.layers.pop(0) #Remove Input Layer
newInput = Input(shape=(480,640,3))    # let us say this new InputLayer
#newOutputs = pretrained_model(newInput)
#newModel = Model(newInput, newOutputs)

premodel.summary()
# Creating dictionary that maps layer names to the layers
layer_dict = dict([(layer.name, layer) for layer in premodel.layers])
print(layer_dict)
#Getting output tensor of the last pretrained layer that we want to include
x = layer_dict['block5_pool'].output #block2

x=premodel.layers[-3].output
flatten=Flatten()(x)
dropout1=Dropout(0.5)(flatten)
dense1=Dense(1,activation='relu')(dropout1)
dense2=Dense(307200,activation='linear')(dense1)
model = Model(input=premodel.input, output=dense2)
model.compile(loss='mean_squared_error', optimizer=Adam(lr = 1e-4)) #lr = 1e-4

#model=pspnet_50_ADE_20K()

out = pretrained_model.predict_segmentation(
    inp=r"G:\Documents\NYU Depth Dataset\nyu_data\X_rgb\rgb_68.png",
    out_fname="out.png"
)