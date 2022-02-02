import cv2
import numpy as np
import time as t
import math
import statistics


from numba import njit, prange, jit
    
@njit(nogil=True, fastmath=True)
def superpixel(labels, num_of_superpixels_result, width, height, Depth_stereo, Depth_pred):
    '''
    1. Convert 'labels' to 'x&y coordinate list'
    '''
    # numba unsupported unknown data type
    x_list = [[np.int64(-1)] for x in range(num_of_superpixels_result)]
    y_list = [[np.int64(-1)] for x in range(num_of_superpixels_result)]
    
    for row in range(width):
        for col in range(height):
            x_list[labels[col][row]].append(row)
            y_list[labels[col][row]].append(col)
            
    for i in range(num_of_superpixels_result-1,-1,-1):
        x_list[i].pop(0)
        y_list[i].pop(0)
        
        xL = len(x_list[i])
        yL = len(y_list[i])
        if xL==0 or yL==0:
            x_list.pop(i)
            y_list.pop(i)
            num_of_superpixels_result = num_of_superpixels_result - 1
            
            
    '''
    2. Distance-based centroid within fragments
    '''
    MDE_list    = [[np.float64(-1)] for x in range(num_of_superpixels_result)]
    stereo_list = [[np.float64(-1)] for x in range(num_of_superpixels_result)]
    distance_list = [[np.float64(-1)] for x in range(num_of_superpixels_result)]
    min_dist_index_num = np.zeros((num_of_superpixels_result), dtype=np.int16)
    # seg_center = [x_pos, y_pos, stereo_depth, MDE_depth, fragment_size] array
    seg_center = np.zeros((num_of_superpixels_result, 5))    
    
    for i in range(num_of_superpixels_result):
        L = len(x_list[i])
        
        sum_x = 0
        sum_y = 0
        
        for j in range(L):
            sum_x += x_list[i][j]
            sum_y += y_list[i][j]
            
        avg_x = sum_x/L
        avg_y = sum_y/L
        
        for j in range(L):
            # Depth within fragment
            MDE_list[i].append(Depth_pred[y_list[i][j],x_list[i][j]])
            stereo_list[i].append(Depth_stereo[y_list[i][j],x_list[i][j]])
            # Distance within fragment's centroid
            distance_list[i].append( (x_list[i][j]-avg_x)**2 + (y_list[i][j]-avg_y)**2 )
        
        MDE_list[i].pop(0)
        stereo_list[i].pop(0)
        distance_list[i].pop(0)
        
        min_dist_index_num[i] = distance_list[i].index(min(distance_list[i]))

        # Output Data Generation
        MDE_np = np.array(MDE_list[i])
        median_MDE = np.median(MDE_np)
        stereo_np = np.array(stereo_list[i])
        median_stereo = np.median(stereo_np)
        x_pos = x_list[i][min_dist_index_num[i]]
        y_pos = y_list[i][min_dist_index_num[i]]
        
        seg_center[i] = [x_pos, y_pos, median_stereo, median_MDE, L]
    
    '''
    3. scaling
    '''
    seg_count = 0
    stereo_sum = 0
    pred_sum = 0
    for i in range(num_of_superpixels_result):
        max_val = 0
        min_val = 1000
        
        L = len(x_list[i])
        for j in range(L):
            x_pos = x_list[i][j]
            y_pos = y_list[i][j]
            stereo_val = Depth_stereo[y_pos,x_pos]
            pred_val = Depth_pred[y_pos,x_pos]
            
            if stereo_val<min_val:
                min_val=stereo_val
            if stereo_val>max_val:
                max_val=stereo_val

        # no support for numpy funtion (delete,all, any, sum etc) 
        # condition of scaling. 1) 
        if 1.5*min_val>max_val and min_val != 0 and min_val>1 and max_val<10:
            stereo_sum += stereo_val
            pred_sum += pred_val
            # seg_count+=1
            
    if (pred_sum!=0 and stereo_sum!=0):
        scaling_factor = stereo_sum/pred_sum
    else:
        scaling_factor = 1
            
    return seg_center, scaling_factor


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
        self.seg_center, scaling_factor = superpixel(labels, num_of_superpixels_result, self.width, self.height, self.Depth_stereo, self.Depth_pred)
        
        return self.seg_center, scaling_factor