# -*- coding: utf-8 -*-
"""
Script for creating depth prediction images from multiple models for comparison
"""
from utils import deep_utils
from utils import image_utils
import numpy as np

#Load models (yaml/json)
models={'unet':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\Final Results\200 Epochs\U-Net\depth_estimation_unet_nyu_model.yaml",
        'wnet_c':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\Final Results\200 Epochs\W-Net Connected\depth_estimation_wnetc_nyu_model.yaml"
        }
#Load weights (h5)
weights={'unet':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\Final Results\200 Epochs\U-Net\depth_estimation_unet_nyu_model.h5",
        'wnet_c':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\Final Results\200 Epochs\W-Net Connected\depth_estimation_wnetc_nyu_model.h5"
         }

#images=[r"G:\Documents\NYU Depth Dataset\nyu_data\X_rgb\rgb_14.png", #9
#        r"G:\Documents\NYU Depth Dataset\nyu_data\X_rgb\rgb_330.png", #533
#        r"G:\Documents\NYU Depth Dataset\nyu_data\X_rgb\rgb_579.png", #661
#        r"G:\Documents\NYU Depth Dataset\nyu_data\X_rgb\rgb_756.png", #854
#        r"G:\Documents\NYU Depth Dataset\nyu_data\X_rgb\rgb_999.png"] #1204

images=[r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\Final Results\200 Epochs\Prediction Images\new_test_images\resize\1.png",
        r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\Final Results\200 Epochs\Prediction Images\new_test_images\resize\2.png",
        r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\Final Results\200 Epochs\Prediction Images\new_test_images\resize\3.png",
        r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\Final Results\200 Epochs\Prediction Images\new_test_images\resize\4.png",
        r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\Final Results\200 Epochs\Prediction Images\new_test_images\resize\5.png"]

for name in models.keys():
    #Load model
    model=deep_utils.load_model(models[name],weights[name])

    for i in range(len(images)):
        #Read test image
        image=image_utils.rgb_read(images[i]) #640x480
        image=image.reshape(1,480,640,3)
        image=np.divide(image,255).astype(np.float16)
        #Predict depth
        y_est=model.predict(image)
        y_est=y_est.reshape((480,640))*255 #De-normalize for depth viewing
        #Save results
        image_utils.heatmap(y_est,save=True,name=f'Image{i}_{name}_gray',cmap='gray')
        image_utils.heatmap(y_est,save=True,name=f'Image{i}_{name}_plasma',cmap='plasma')
