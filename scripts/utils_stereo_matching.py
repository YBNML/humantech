#!/usr/bin/env python3

import numpy as np
import cv2
from scipy.ndimage.filters import *
from scipy.sparse import diags
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
from numba import jit, prange



@jit(nopython = True, parallel = True, cache = True)
def zncc(left_image, right_image, D, R, adaptive):
    (H,W) = left_image.shape
    cv    = np.ones((D,H,W)) * 1000     # Cost_Volume's init value : MAX Value(=1000)
    
    # Loop over image
    for y in prange(R, H - R):
        # Adaptive Window Size (=tempR)
        tempR = R
        if 4<y and H-5>y:
            if adaptive[y] <1.8:
                tempR=R+2
            elif adaptive[y] >6:
                tempR=R-1
        # Pixel number of window size
        N = (2*tempR+1)**2
        
        for x in prange(tempR, W - tempR):
            
            l_mean = 0
            for i in range(-n, n+1):
                for j in range(-n, n+1):
                    l_mean += img[u+i][v+j]
            # Loop over all possible disparities
            for d in prange(0, D):
                if x-d-tempR<0:
                    break
                
                l_mean = 0
                for i in range(-n, n+1):
                    for j in range(-n, n+1):
                        l_mean += img[u+i][v+j]
                
            

def get_f(D, L1 = 0.025, L2 = 0.5):
    """
    Get pairwise cost matrix for semi-global matching
    
        @param D:  disparities, number of possible choices
        @param L1: parameter for setting cost for jumps between two layers of depth
        @param L2: cost for jumping more than one layer of depth
    
        @return: pairwise_costs of shape (D,D)
    """
    
    return np.full((D, D), L2) + diags([L1 - L2, -L2, L1 - L2], [-1, 0, 1], (D, D)).toarray()


@jit(nopython = True, parallel = True, cache = True)
def compute_costvolume_zncc_test2(left_image, right_image, D, R, adaptive):
    assert(left_image.shape == right_image.shape)
    assert(D > 0)
    assert(R > 0)

    (H,W) = left_image.shape
    cv    = np.ones((D,H,W)) * 1000      # Cost Volume
    
    # Loop over image
    for y in prange(R, H - R):
        # Adaptive Window Size (=tempR)
        tempR = R
        if 4<y and H-5>y:
            if adaptive[y] <1.8:
                tempR=R+2
            elif adaptive[y] >6:
                tempR=R-1
                
        N = (2*tempR+1)**2
        
        for x in prange(tempR, W - tempR):
            l_window    = left_image[y-tempR:y+tempR+1,x-tempR:x+tempR+1]
            l_mean      = np.sum(l_window)
            l_mean      = l_mean/N
            l           = l_window - l_mean
            l_var       = np.sum(l**2)
            # Loop over all possible disparities
            for d in prange(0, D):
                if x-d-tempR<0:
                    break
                r_window    = right_image[y-tempR:y+tempR+1,x-d-tempR:x-d+tempR+1]
                r_mean      = np.sum(r_window)
                r_mean      = r_mean/N
                r           = r_window - r_mean
                r_var       = np.sum(r**2)
                
                l_r         = np.sum(l*r)
                
                cv[d,y,x] = -l_r/np.sqrt(l_var*r_var)
                
    # return np.transpose(cv, (1, 2, 0))

@jit(nopython = True,  parallel = True, cache = True)
def compute_costvolume_zncc_left(left_image, right_image, D, R, adaptive):
    """
    Compute a cost volume with maximum disparity D considering a neighbourhood R with Normalized Cross Correlation (NCC)
    
        @param left_image:  left input image of size (H,W)
        @param right_image: right input image of size (H,W)
        @param D:           maximum disparity
        @param radius:      radius of the filter
        
        @return:            cost volume of size (H,W,D)
    """

    assert(left_image.shape == right_image.shape)
    assert(D > 0)
    assert(R > 0)

    (H,W) = left_image.shape
    cv    = np.zeros((D,H,W))
    
    # Loop over image
    for y in prange(R, H - R):
        # print("#1 ",y)
        tempR = R
        if 2<<y and H-2>y:
            if adaptive[y] <1.8:
                tempR=R+2
            elif adaptive[y] >6:
                tempR=R-1

        for x in prange(tempR, W - tempR):
            
            # Loop over all possible disparities
            for d in prange(0, D):
                
                l_mean = 0
                r_mean = 0
                n      = 0
                
                l_mean = np.sum(left_image[y-tempR:y+tempR+1,x-tempR:x+tempR+1])
                r_mean = np.sum(right_image[y-tempR:y+tempR+1,x-tempR:x+tempR+1])
                N = (2*tempR+1)**2
                l_mean = l_mean/N
                r_mean = l_mean/N
                
                l_r   = 0
                l_var = 0
                r_var = 0
            
                for v in range(-tempR, tempR + 1):
                    for u in range(-tempR, tempR + 1):
                        
                        # Calculate terms
                        l = left_image[y+v, x+u+d]    - l_mean
                        r = right_image[y+v, x+u] - r_mean
                        
                        l_r   += l*r
                        l_var += l**2
                        r_var += r**2

                # Assemble terms
                cv[d,y,x] = -l_r/np.sqrt(l_var*r_var)
    
    return np.transpose(cv, (1, 2, 0))


def compute_wta(cv):
    '''
    Compute the best disparity on a scan line using winner-takes-it-all
        
        @param cv: a given cost volume (H,W,D)
        
        @return    a disparity image (H,W)
    '''
    
    assert(cv.ndim == 3)
    return np.argmin(cv, axis=2)


def get_f(D, L1 = 0.025, L2 = 0.5):
    """
    Get pairwise cost matrix for semi-global matching
    
        @param D:  disparities, number of possible choices
        @param L1: parameter for setting cost for jumps between two layers of depth
        @param L2: cost for jumping more than one layer of depth
    
        @return: pairwise_costs of shape (D,D)
    """
    
    return np.full((D, D), L2) + diags([L1 - L2, -L2, L1 - L2], [-1, 0, 1], (D, D)).toarray()


@jit(nopython = True, parallel = True, cache = True)
def compute_message(cv, f):
    """
    Compute the messages in one particular direction for semi-global matching
    
        @param cv: cost volume of shape (H,W,D)
        @param f:  pairwise costs of shape (D,D)
    
        @return:   messages for all H in positive direction of W with possible options D (H,W,D)
    """
    
    (H,W,D) = cv.shape
    mes     = np.zeros((H,W,D))
    
    # Loop over passive direction
    for y in prange(0, H):
        
        # Loop over forward direction
        for x in range(0, W - 1):
            
            # Loop over all possible nodes
            for t in range(0, D):
                
                # Loop over all possible connections
                buffer = np.zeros(D)
                for s in prange(0, D):
                    # Input messages + unary cost + binary cost
                    buffer[s] = mes[y,x,s] + cv[y,x,s] + f[t,s]
                
                # Choose path of least effort
                mes[y, x+1, t] = np.min(buffer)
                
    return mes
    

def compute_sgm(cv, f):
    """
    Compute semi-global matching by message passing in four directions
    
        @param cv: cost volume of shape (H,W,D)
        @param f:  pairwise costs of shape (H,W,D,D)
    
        @return:   pixel-wise disparity map of shape (H,W)
    """
    # Messages for every single spatial direction and collect in single message
    (H,W,D) = cv.shape
    mes     = np.zeros((H,W,D))
    
    # Positive W
    mes += compute_message(cv, f)
    
    # Negative W
    mes_buffer  = np.zeros((H,W))
    mes_buffer  = compute_message(np.flip(cv, axis=1), f)
    mes        += np.flip(mes_buffer, axis=1)
    
    # Positive H
    mes_buffer  = compute_message(np.transpose(cv, (1, 0, 2)), f)
    mes        += np.transpose(mes_buffer, (1, 0, 2))
    
    # Negative H
    mes_buffer  = compute_message(np.flip(np.transpose(cv, (1, 0, 2)), axis=1), f)
    mes        += np.transpose(np.flip(mes_buffer, axis=1), (1, 0, 2))
    
    # Choose best believe from all messages
    disp_map = np.zeros((H,W))
    for y in range(0, H):
        for x in range(0, W):
            # Minimum argument of unary cost and messages
            disp_map[y,x] = np.argmin(cv[y,x,:] + mes[y,x,:])
    
    return disp_map