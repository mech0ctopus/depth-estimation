# -*- coding: utf-8 -*-
"""
Script for creating depth prediction images from multiple models for comparison
"""
from utils import deep_utils
from utils import image_utils
import numpy as np
from os.path import basename
from models import models
from tensorflow.keras.optimizers import Adam

#Load weights (h5)
weights={'unet':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Weights & Models\20Epochs_NoAugment_LowerLR\20200213-065621\U-Net_weights_best.hdf5",
        'wnet':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Weights & Models\20Epochs_NoAugment_LowerLR\20200213-082137\W-Net_weights_best.hdf5",
        'wnet_c':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Weights & Models\20Epochs_NoAugment_LowerLR\20200212-205108\W-Net_Connected_weights_best.hdf5"
         }

images=[r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Graphs & Pictures\LR_0.0001\MoreCars\5.jpg"]

model_name='wnet'
model=models.wnet()

model.compile(loss='mean_squared_error',optimizer=Adam(),metrics=['mse']) 
model.load_weights(weights['wnet'])


for i in range(len(images)):
    #Read test image
    image=image_utils.rgb_read(images[i]) #640x480
    image=image.reshape(1,480,640,3)
    image=np.divide(image,255).astype(np.float16)
    image_name=basename(images[i]).split('.')[0]
    #Predict depth
    y_est=model.predict(image)
    y_est=y_est.reshape((480,640))*255 #De-normalize for depth viewing
    #Save results
    #image_utils.heatmap(y_est,save=True,name=f'{image_name}_{model_name}_gray',cmap='gray')
    image_utils.heatmap(y_est,save=True,name=f'{image_name}_{model_name}_plasma',cmap='plasma')
