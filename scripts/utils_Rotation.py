#!/usr/bin/env python3

'''
Rotation Matrix
'''

import math
import time as t
import numpy as np

from numba import jit, prange

from utils_Parameter import parameter


param = parameter()

fx = param.get_fx()
fy = param.get_fy()
ppx = param.get_px()
ppy = param.get_fy()
D = [1e-08, 1e-08, 1e-08, 1e-08, 1e-08]

@jit(nopython = True,  parallel = True, cache = True)
def rotate_data(left_data, right_data):
    # if not, end of the process over.
    assert(left_data.shape == right_data.shape)
    H,W = right_data.shape
    
    for ih in range(H):
        for iw in range(W):
            l_depth = left_data[ih,iw]
            r_depth = right_data[ih,iw]
            
            # deproject_pixel_to_point
            x = (iw - ppx) / fx
            y = (ih - ppy) / fy

            r2 = x*x + y*y
            
            f = 1
            ux = x*f
            uy = y*f
            
            point_x     = l_depth * ux
            point_y     = l_depth * uy
            point_depth = l_depth
            
            l_depth = 0.2588190451*point_x + 0.96592582628*point_depth
            left_data[ih,iw] = l_depth
            
            point_x     = r_depth * ux
            point_y     = r_depth * uy
            point_depth = r_depth
            
            r_depth = -0.2588190451*point_x + 0.96592582628*point_depth
            right_data[ih,iw] = r_depth

    return left_data, right_data