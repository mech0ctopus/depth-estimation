# -*- coding: utf-8 -*-
"""
Assumes model and X_test are loaded as variables.
"""
from timeit import default_timer as timer
from models import models
from tensorflow.keras.optimizers import Adam
from utils import image_utils
import numpy as np

#Load weights (h5)
weights={'unet':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Weights & Models\20Epochs_NoAugment_LowerLR\20200213-065621\U-Net_weights_best.hdf5",
        'wnet':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Weights & Models\20Epochs_NoAugment_LowerLR\20200213-082137\W-Net_weights_best.hdf5",
        'wnet_c':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Weights & Models\20Epochs_NoAugment_LowerLR\20200212-205108\W-Net_Connected_weights_best.hdf5"
         }

model=models.pretrained_unet()
model.compile(loss='mean_squared_error',optimizer=Adam(),metrics=['mse']) 
model.load_weights(weights['unet'])

times=[]

#Read test image
test_image=r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Graphs & Pictures\LR_0.0001\CityImages\1.jpg"
test_image=image_utils.rgb_read(test_image) #640x480
test_image=test_image.reshape(1,480,640,3)
test_image=np.divide(test_image,255)

for i in range(101):  
    start_time = timer()
    y_est=model.predict(test_image)
    end_time = timer()
    if i!=0:
        times.append(end_time-start_time)
        print(end_time-start_time)
    
print('Average (ms): '+str(sum(times)/len(times)))
print('Max. (ms): '+str(max(times)))
print('Min. (ms): '+str(min(times)))