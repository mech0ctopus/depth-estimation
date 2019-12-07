# -*- coding: utf-8 -*-
"""
Create and save visualizations of convolutional layers.  Can use random input
or specify an image file.

Original file from:
https://github.com/jacobgil/keras-filter-visualization/issues/1
"""

from keras import backend as K
from deep_utils import load_model
import numpy as np
import cv2
from tqdm import tqdm

K.set_learning_phase(1)
#Configuration:
img_width, img_height = 480, 640
input_shape = (img_width, img_height, 3)
num_filters = 9
iterations = 10
weights_path = r"C:\Users\Craig\Documents\GitHub\depth-estimation\TrainedModels\resnet_unet\depth_estimation_cnn_nyu_model_Round7.h5"
#img_path = r"G:\Documents\NYU Depth Dataset\nyu_data\X_rgb\rgb_911.png"
img_path=None
filter_indexes = range(0, num_filters)

def save_filters(filters, img_width, img_height, layer_name):
    margin = 5
    n = int(len(filters)**0.5)
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            index = i * n + j
            if index < len(filters):
                img = filters[i * n + j]
                stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                                 (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    cv2.imwrite('depth_estimation_'+f'{layer_name}'+'_stitched_filters_%dx%d.png' % (n, n), stitched_filters)

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    #layer_dict = dict([(layer.name, layer) for layer in model.layers])
    #layer_output = layer_dict[layer_name].output
    layer_output = model.get_layer(layer_name).output
    return layer_output

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

#Define regularizations:
def blur_regularization(img, grads, size = (3, 3)):
    return cv2.blur(img, size)

def decay_regularization(img, grads, decay = 0.8):
    return decay * img

def clip_weak_pixel_regularization(img, grads, percentile = 1):
    clipped = img
    threshold = np.percentile(np.abs(img), percentile)
    clipped[np.where(np.abs(img) < threshold)] = 0
    return clipped

def gradient_ascent_iteration(loss_function, img, lr=0.9):
    loss_value, grads_value = loss_function([img])    
    gradient_ascent_step = img + grads_value * lr

    #Convert to row major format for using opencv routines
    grads_row_major = grads_value[0, :]
    img_row_major = gradient_ascent_step[0, :]

    #List of regularization functions to use
    regularizations = [blur_regularization, decay_regularization, clip_weak_pixel_regularization]

    #The reguarlization weights
    weights = np.float32([3, 3, 1])
    weights /= np.sum(weights)

    images = [reg_func(img_row_major, grads_row_major) for reg_func in regularizations]

    #Craig Edit
    for idx, image in enumerate(images):
        if image.shape==(480,640,1):
            image=image[:,:,0]
            images[idx]=image
    #End

    weighted_images = np.float32([w * image for w, image in zip(weights, images)])
    img = np.sum(weighted_images, axis = 0)

    #Convert image back to 1 x 3 x height x width
    img = np.float32([img])

    return img

def visualize_filter(input_img, filter_index, img_placeholder, layer, number_of_iterations = 20):
    loss = K.mean(layer[:, :, :, filter_index])
    grads = K.gradients(loss, img_placeholder)[0]
    grads = normalize(grads)
    # this function returns the loss and grads given the input picture
    iterate = K.function([img_placeholder], [loss, grads])

    img = input_img * 1

    # we run gradient ascent for 20 steps
    for i in range(number_of_iterations):
        img = gradient_ascent_iteration(iterate, img)

    # decode the resulting input image
    img = deprocess_image(img[0])
    #print("Done with filter", filter_index)
    return img

#model = get_model(input_shape)
model=load_model(r"C:\Users\Craig\Documents\GitHub\depth-estimation\TrainedModels\resnet_unet\depth_estimation_cnn_nyu_model_Round7.yaml",
                 r"C:\Users\Craig\Documents\GitHub\depth-estimation\TrainedModels\resnet_unet\depth_estimation_cnn_nyu_model_Round7.h5")
layer_names=[layer.name for layer in model.layers]
layer_names=[layer_name for layer_name in layer_names if 'conv' in layer_name]
model.load_weights(weights_path)
input_placeholder = model.input

for layer_name in layer_names:
    print(layer_name)
    layer = get_output_layer(model, layer_name)
    #print(model.input)
    if img_path is None:
        # we start from a gray image with some random noise
        init_img = np.random.random((1, img_width, img_height, 3)) * 20 + 32.
    else:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_height,img_width))
        init_img = [img]
    
    vizualizations = [None] * len(filter_indexes)
    for i in tqdm(range(len(filter_indexes))):
        #for i,  in enumerate(filter_indexes):
        index = filter_indexes[i]
        vizualizations[i] = visualize_filter(init_img, index, input_placeholder,layer, iterations)
        #Save the visualizations see the progress made so far
        save_filters(vizualizations, img_width, img_height,layer_name)
print('Done.')