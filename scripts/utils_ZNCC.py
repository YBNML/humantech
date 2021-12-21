#!/usr/bin/env python3

import numpy as np

from numba import njit, prange, jit


@njit(nogil=True, parallel = True, fastmath=True)
def zncc_left(left_image, right_image, D, R, adaptive):
    (H,W)   = left_image.shape
    cv    = np.ones((H,W))*200      # disparity image
    
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
            # left_window's average
            l_avg = 0  
            for v in range(-tempR, tempR + 1):
                for u in range(-tempR, tempR + 1):
                    l_avg += left_image[y+v, x+u]
            l_avg = l_avg/N
            # left_window's standard_deviation
            l_dev = 0
            for v in range(-tempR, tempR + 1):
                for u in range(-tempR, tempR + 1):
                    l_dev += (left_image[y+v, x+u] - l_avg)**2
            l_dev = (l_dev**0.5)/(2*tempR+1)
            
            
            # # Minimum cost value
            min_cost = 10000
            min_disparity = 200
            
            # Considering disparity range
            for d in range(1, D):
                if x-d-tempR<0:
                    break
                
                # right_window's average
                r_avg = 0  
                for v in range(-tempR, tempR + 1):
                    for u in range(-tempR, tempR + 1):
                        r_avg += right_image[y+v, x+u-d]
                r_avg = r_avg/N
                # right_window's standard_deviation
                r_dev = 0
                for v in range(-tempR, tempR + 1):
                    for u in range(-tempR, tempR + 1):
                        r_dev += (right_image[y+v, x+u-d] - r_avg)**2
                r_dev = (r_dev**0.5)/(2*tempR+1)
                
                # Store cost
                temp_cost = 0
                for v in range(-tempR, tempR + 1):
                    for u in range(-tempR, tempR + 1):
                        temp_cost += (left_image[y+v, x+u] - l_avg)*(right_image[y+v, x+u-d] - r_avg)
                
                if temp_cost < min_cost:
                    min_cost = temp_cost
                    min_disparity = d
                    
            cv[y,x] = min_disparity
            
    return cv
    
    
@njit(nogil=True, parallel = True, fastmath=True)
def zncc_right(left_image, right_image, D, R, adaptive):
    (H,W)   = right_image.shape
    cv    = np.ones((H,W))*200      # disparity image
    
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
            # right_window's average
            r_avg = 0  
            for v in range(-tempR, tempR + 1):
                for u in range(-tempR, tempR + 1):
                    r_avg += right_image[y+v, x+u]
            r_avg = r_avg/N
            # right_window's standard_deviation
            r_dev = 0
            for v in range(-tempR, tempR + 1):
                for u in range(-tempR, tempR + 1):
                    r_dev += (right_image[y+v, x+u] - r_avg)**2
            r_dev = (r_dev**0.5)/(2*tempR+1)
                
            # Minimum cost value
            min_cost = 5000
            min_disparity = 200
            
            # Considering disparity range
            for d in prange(1, D):
                if x+d+tempR>=640:
                    break
                
                # left_window's average
                l_avg = 0  
                for v in range(-tempR, tempR + 1):
                    for u in range(-tempR, tempR + 1):
                        l_avg += left_image[y+v, x+u+d]
                l_avg = l_avg/N
                # left_window's standard_deviation
                l_dev = 0
                for v in range(-tempR, tempR + 1):
                    for u in range(-tempR, tempR + 1):
                        l_dev += (left_image[y+v, x+u+d] - l_avg)**2
                l_dev = (l_dev**0.5)/(2*tempR+1)
                
                # Store cost
                temp_cost = 0
                for v in range(-tempR, tempR + 1):
                    for u in range(-tempR, tempR + 1):
                        temp_cost += (left_image[y+v, x+u+d] - l_avg)*(right_image[y+v, x+u] - r_avg)
                
                if temp_cost < min_cost:
                    min_cost = temp_cost
                    min_disparity = d
                
            cv[y,x] = min_disparity
            
    return cv
