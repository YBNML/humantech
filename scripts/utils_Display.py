#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import skimage
from skimage.transform import resize

def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:,:,0]
    return np.stack((i,i,i), axis=2)

def DISPLAY(image_data):
    if len(image_data.shape) < 4: image_data = image_data.reshape((1, image_data.shape[0], image_data.shape[1], 1))

    image_viz = display_images(image_data)
    r, g, b = cv2.split(image_viz)
    image_viz = cv2.merge((b, g, r))
    
    width = image_data.shape[2]
    height = image_data.shape[1]
    image_viz = cv2.resize(image_viz, (width, height), interpolation=cv2.INTER_AREA)

    if len(image_data.shape) == 4: image_data = image_data.reshape((image_data.shape[1], image_data.shape[2]))

    width = image_data.shape[1]
    height = image_data.shape[0]
    image_data = cv2.resize(image_data, (width, height), interpolation=cv2.INTER_AREA)

    return image_data, image_viz

def display_images(outputs, inputs=None, gt=None, is_colormap=True, is_rescale=True):
    plasma = plt.get_cmap('plasma')

    shape = (outputs[0].shape[0], outputs[0].shape[1], 3)
    all_images = []

    for i in range(outputs.shape[0]):
        imgs = []
        if isinstance(inputs, (list, tuple, np.ndarray)):
            x = to_multichannel(inputs[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        if isinstance(gt, (list, tuple, np.ndarray)):
            x = to_multichannel(gt[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        if is_colormap:
            rescaled = outputs[i][:,:,0]
            if is_rescale:
                rescaled = rescaled - np.min(rescaled)
                rescaled = rescaled / np.max(rescaled)
            imgs.append(plasma(rescaled)[:,:,:3])
        else:
            imgs.append(to_multichannel(outputs[i]))

        img_set = np.hstack(imgs)
        all_images.append(img_set)

    all_images = np.stack(all_images)

    return skimage.util.montage(all_images, multichannel=True, fill=(0,0,0))