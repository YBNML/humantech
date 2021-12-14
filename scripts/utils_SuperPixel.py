import cv2
import numpy as np
import time as t
import math
import statistics


from numba import njit, prange, jit

@njit
def labels_to_xy(labels, num_of_superpixels_result, width, height):
    # numba unsupported unknown data type
    x_list = [[np.int64(-1)] for x in range(num_of_superpixels_result)]
    y_list = [[np.int64(-1)] for x in range(num_of_superpixels_result)]
    
    for row in range(width):
        for col in range(height):
            x_list[labels[col][row]].append(row)
            y_list[labels[col][row]].append(col)

    # print(x_list)
    for i in range(num_of_superpixels_result):
        x_list[i].pop(0)
        y_list[i].pop(0)
        

class SuperPixelSampler:
    def  __init__(self):
    	# set parameters for superpixel segmentation
        self.num_superpixels = 2000  # desired number of superpixels
        self.num_iterations = 4     # number of pixel level iterations. The higher, the better quality
        self.prior = 2              # for shape smoothing term. must be [0, 5]
        self.num_levels = 4
        self.num_histogram_bins = 5 # number of histogram bins

    def superPixel(self, img, stereo, pred):
        # STEP1 : Image load and Option setting
        self.RGB_im = img    
        self.Depth_stereo = stereo
        self.Depth_pred = pred
        self.height, self.width, self.channels = self.RGB_im.shape

        # initialize SEEDS algorithm
        self.seeds = cv2.ximgproc.createSuperpixelSEEDS(self.width, self.height, self.channels, self.num_superpixels, self.num_levels, self.prior, self.num_histogram_bins)
        # run SEEDS
        self.seeds.iterate(self.RGB_im, self.num_iterations)
        # retrieve the segmentation result.
        # height x width matrix. Each component indicates the superpixel index of the corresponding pixel position
        labels = self.seeds.getLabels() 
        # get number of superpixel's group(slice)
        num_of_superpixels_result = self.seeds.getNumberOfSuperpixels()

        # Convert 'labels' to 'x&y coordinate list'
        st = t.time()
        labels_to_xy(labels, num_of_superpixels_result, self.width, self.height)
        et = t.time()
        print(et-st)
        