# -*- coding: utf-8 -*-
"""
Image depth functions.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def heatmap(image_path):
    '''Plots heatmap of depth data from image.'''
    pic=Image.open(image_path)
    pic_array=np.array(pic)
    plt.imshow(pic_array, cmap='plasma', interpolation='nearest')
    plt.show()
    
if __name__=='__main__':
    heatmap(r"D:\DeepLearningData\bag_example_pics\Multisense-depth\image_94.jpeg")
