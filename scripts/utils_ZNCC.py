#!/usr/bin/env python3

import numpy as np

from numba import njit, jit, prange

cv_result    = np.zeros((400,360))

@njit(nopython = True, parallel = True, cache = True, fastmath=True)
def zncc(left_image, right_image, D, R, adaptive):
    (H,W)   = left_image.shape
    cv    = np.zeros((D,H,W))
    # cost  = [1000 for col in range(D)]
    
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
            
            # Considering disparity range
            for d in prange(0, D):
                if x-d-tempR<0:
                    break
                
                # # Minimum cost value
                min_cost = 5000
                min_cost_disp = -1
                
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
                
                # Store cost
                sum_arr = 0
                for v in range(-tempR, tempR + 1):
                    for u in range(-tempR, tempR + 1):
                        sum_arr += (left_image[y+v, x+u] - l_avg)*(right_image[y+v, x+u] - r_avg)
                
                temp = sum_arr/(N * l_dev * r_dev)
                cv[d,y,x] = temp
            #     if temp < min_cost:
            #         min_cost = temp;
            #         min_cost_disp = d
                    
            # cv[y,x] = min_cost_disp
    
    # temp_cv = cv_img.copy()     
    return 0
    