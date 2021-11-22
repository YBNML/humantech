#!/usr/bin/env python3

import os
import sys

from pathlib import Path

FILE = Path(__file__).resolve()
ROS_WS = FILE.parents[2]  # 2단계 상위 폴더의 경로


class Link_Adabins:
    def __init__(self):
        # Insert Adabins's path 
        adabins_path = os.path.abspath(os.path.join(ROS_WS, 'adabins')) + "/scripts"
        sys.path.append(adabins_path) # utils 절대경로를 환경변수에 추가
        from utils_Adabins import Adabins
        
        self.adabins = Adabins()
    def predict(self):
        # Left
        centers, left_adabins_output = self.adabins.predict_pil(left_adabins_input)
        self.left_depth = left_adabins_output.squeeze()


if __name__=="__main__":
    adabins = Link_Adabins()