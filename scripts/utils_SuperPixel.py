import cv2
import numpy as np
import time as t
import math
import statistics

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
        # get number of superpixel
        num_of_superpixels_result = self.seeds.getNumberOfSuperpixels()

        # label by segmentation
        st = t.time()
        res_list = [[] for i in range(num_of_superpixels_result)]
        for row in range(self.width):
            for col in range(self.height):
                res_list[labels[col][row]].append([col,row])
        self.label_list = res_list

        # average by segmentation
        distance_list = [[] for i in range(num_of_superpixels_result)]
        min_dist_num = np.zeros((num_of_superpixels_result), dtype=np.int)
        distance_median_pred = [[] for i in range(num_of_superpixels_result)]
        distance_median_stereo = [[] for i in range(num_of_superpixels_result)]

        for i in range(num_of_superpixels_result):
            sum_col = sum(res_list[i][:][0])
            sum_row = sum(res_list[i][:][1])

            L = len(res_list[i]) # segmentation당 pixel 수
            avg_col = int(np.round(sum_col/L))
            avg_row = int(np.round(sum_row/L))

            # segmentation의 중심에서부터의 거리
            for j in range(L):
                distance_median_pred[i].append(self.Depth_pred[res_list[i][j][0],res_list[i][j][1]])
                distance_median_stereo[i].append(self.Depth_stereo[res_list[i][j][0],res_list[i][j][1]])
                distance_list[i].append(math.pow(res_list[i][j][0]-avg_col,2)+math.pow(res_list[i][j][1]-avg_row,2))
            min_dist_num[i]=(distance_list[i].index(min(distance_list[i])))
        # print(t.time()-st)
        
        # segmentation의 중심을 구하기
        self.seg_center = np.zeros((num_of_superpixels_result, 5))
        for i in range(num_of_superpixels_result):
            median_pred = statistics.median(distance_median_pred[i])
            median_stereo = statistics.median(distance_median_stereo[i])
            
            col_pos = res_list[i][min_dist_num[i]][0]
            row_pos = res_list[i][min_dist_num[i]][1]

            self.seg_center[i] = [col_pos, row_pos, median_stereo, median_pred, len(res_list[i])]

    '''
    output = 좌표, 좌표, 깊이(좌표값), seg 갯수
    '''
    def get_seg_center(self):
        return self.seg_center

    '''
    성능을 이미지로서 보여줌
    '''
    def remove_outlier(self):
        # self.Depth_pred[res_list[i][j][0],res_list[i][j][1]])
        temp_seg_center = self.seg_center
        num_of_superpixels_result = self.seeds.getNumberOfSuperpixels()
        del_num = 0
        for i in range(num_of_superpixels_result):
            # rng = 0
            max=0
            min=1000
            L = len(self.label_list[i])
            for j in range(L):
                H = self.label_list[i][j][0]
                W = self.label_list[i][j][1]

                # if 440>W or 840<W or 35>H or 685<H:
                    # rng+=1
                val = self.Depth_stereo[H,W]

                if val<min:
                    min=val
                if val>max:
                    max=val
            # if 1.5*min<max or rng>0:
            if 1.5*min<max or min == 0:
                temp_seg_center = np.delete(temp_seg_center, (i-del_num), axis=0)
                del_num+=1

        # for i in range(len(temp_seg_center)):
            # print(temp_seg_center[i][0],'\t',temp_seg_center[i][1],'\t',temp_seg_center[i][2],'\t',temp_seg_center[i][3])
        # print(num_of_superpixels_result)
        self.seg_center = temp_seg_center
                
                

    def superPixel2(self, img, pred):
        # STEP1 : Image load and Option setting
        self.RGB_im = img
        self.Depth_pred = pred
        self.height, self.width, self.channels = self.RGB_im.shape

        # initialize SEEDS algorithm
        self.seeds = cv2.ximgproc.createSuperpixelSEEDS(self.width, self.height, self.channels, self.num_superpixels, self.num_levels, self.prior, self.num_histogram_bins)
        # run SEEDS
        self.seeds.iterate(self.RGB_im, self.num_iterations)
        # retrieve the segmentation result.
        # height x width matrix. Each component indicates the superpixel index of the corresponding pixel position
        labels = self.seeds.getLabels() 
        # get number of superpixel
        num_of_superpixels_result = self.seeds.getNumberOfSuperpixels()

        # label by segmentation
        st = t.time()
        res_list = [[] for i in range(num_of_superpixels_result)]
        for row in range(self.width):
            for col in range(self.height):
                res_list[labels[col][row]].append([col,row])
        self.label_list = res_list

        # average by segmentation
        distance_list = [[] for i in range(num_of_superpixels_result)]
        min_dist_num = np.zeros((num_of_superpixels_result), dtype=np.int)

        for i in range(num_of_superpixels_result):
            sum_col = sum(res_list[i][:][0])
            sum_row = sum(res_list[i][:][1])

            L = len(res_list[i]) # segmentation당 pixel 수
            avg_col = int(np.round(sum_col/L))
            avg_row = int(np.round(sum_row/L))

            # segmentation의 중심에서부터의 거리
            for j in range(L):
                distance_list[i].append(math.pow(res_list[i][j][0]-avg_col,2)+math.pow(res_list[i][j][1]-avg_row,2))
            min_dist_num[i]=(distance_list[i].index(min(distance_list[i])))
        # print(t.time()-st)
        
        # segmentation의 중심을 구하기
        self.seg_center2 = np.zeros((num_of_superpixels_result, 4))
        for i in range(num_of_superpixels_result):
            col_pos = res_list[i][min_dist_num[i]][0]
            row_pos = res_list[i][min_dist_num[i]][1]

            self.seg_center2[i] = [col_pos, row_pos, pred[col_pos,row_pos], len(res_list[i])]


    '''
    output = 좌표, 좌표, 깊이(좌표값), seg 갯수
    '''
    def get_seg_center2(self):
        return self.seg_center2

