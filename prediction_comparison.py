# -*- coding: utf-8 -*-
"""
Script for creating depth prediction images from multiple models for comparison
"""
import deep_utils
import image_utils
import numpy as np

#40 epochs
#Load models (yaml/json)
#models={'cnn':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\40 Epochs\CNN\depth_estimation_cnn_nyu_model.yaml",
#        'rcnn':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\40 Epochs\RCNN\depth_estimation_rcnn_nyu_model.yaml",
#        'unet':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\40 Epochs\U-Net\depth_estimation_unet_resnet_nyu_model.yaml",
#        'unet_cnn':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\40 Epochs\U-Net + CNN\depth_estimation_unet+cnn_nyu_model.yaml",
#        'unet_rcnn':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\40 Epochs\U-Net + RCNN\depth_estimation_unet_rcnn_nyu_model.yaml",
#        'wnet':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\40 Epochs\W-Net\depth_estimation_wnet_nyu_model.yaml",
#        'wnet_c':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\40 Epochs\W-Net Connected\depth_estimation_wnetc_nyu_model.yaml"
#        }
##Load weights (h5)
#weights={'cnn':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\40 Epochs\CNN\depth_estimation_cnn_nyu_model.h5",
#         'rcnn':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\40 Epochs\RCNN\depth_estimation_rcnn_nyu_model.h5",
#         'unet':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\40 Epochs\U-Net\depth_estimation_unet_resnet_nyu_model.h5",
#         'unet_cnn':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\40 Epochs\U-Net + CNN\depth_estimation_unet+cnn_nyu_model.h5",
#         'unet_rcnn':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\40 Epochs\U-Net + RCNN\depth_estimation_unet_rcnn_nyu_model.h5",
#         'wnet':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\40 Epochs\W-Net\depth_estimation_wnet_model.h5",
#         'wnet_c':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\40 Epochs\W-Net Connected\depth_estimation_wnetc_nyu_model.h5"
#         }

##200 Epochs
#models={'cnn':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\200 Epochs\CNN\depth_estimation_cnn_nyu_model.yaml",
#        'unet':r,
#        'wnet_c':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\200 Epochs\W-Net Connected\depth_estimation_wnet_connected_nyu_model.yaml"
#        }
##Load weights (h5)
#weights={'cnn':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\200 Epochs\CNN\depth_estimation_cnn_nyu_model.h5",
#        'unet':r,
#        'wnet_c':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\200 Epochs\W-Net Connected\depth_estimation_wnet_connected_nyu_model.h5"
#         }

#640 Epochs
models={'wnet_c':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\640 Epochs\W-Net Connected\depth_estimation_wnetc_nyu_model.yaml"}
#Load weights (h5)
weights={'wnet_c':r"C:\Users\Craig\Documents\GitHub\depth-estimation\Final Results\640 Epochs\W-Net Connected\depth_estimation_wnetc_nyu_model.h5"}

images=[r"G:\Documents\NYU Depth Dataset\nyu_data\X_rgb\rgb_9.png",
        r"G:\Documents\NYU Depth Dataset\nyu_data\X_rgb\rgb_533.png",
        r"G:\Documents\NYU Depth Dataset\nyu_data\X_rgb\rgb_661.png",
        r"G:\Documents\NYU Depth Dataset\nyu_data\X_rgb\rgb_854.png",
        r"G:\Documents\NYU Depth Dataset\nyu_data\X_rgb\rgb_1204.png"]

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
