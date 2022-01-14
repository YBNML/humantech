'''
#include "BehaviourArbitration.h"

The old Kinect has a depth image resolution of 320 x 240 pixels with a fov of 58.5 x 46.6
degrees resulting in an average of about 5 x 5 pixels per degree. (see source 1) << 1 patch = 5x5 pixel

source: http://smeenk.com/kinect-field-of-view-comparison/
'''

import math
import numpy as np
import cv2

from numba import njit


# def BehaviourArbitration():
# Goal = goto
# Obstacle = avoid
lambdaGoalHorz = 0.5;                   # Linear multiplier for the angular velocity
lambdaObstacleHorzNormal = 5;           # 장애물 noraml??, 장애물 인지 계수
lambdaObstacleHorzAggressive = 50;      # 장애물 aggressive??, 장애물 인지 계수
lambdaObstacleHorz = lambdaObstacleHorzNormal # Default value

lambdaObstacleVert = 5

# w, 3페이지에 4번 수식 참고
weightGoalHorz = 0.3
weightObstacleHorz = 0.7

# gain 설정
obstacleDistanceGainHorzNormal = 0.1            # Smaller = more sensitive 
obstacleDistanceGainHorzAggressive = 0.005      # Smaller = more sensitive
obstacleDistanceGainHorz = obstacleDistanceGainHorzNormal  # Default value

# 수평방향 각 최대값 설정
# fov = 82
hfov = 112
vfov = 50
angularRangeHorz = (hfov)*math.pi/180;
angularRangeVert = (vfov)*math.pi/180;

# 일단 수평에 대해서만 고려를 할거라 생략
# lambdaObstacleVert = 5
# angularRangeVert = 27.5*math.pi/180
# obstacleDistanceGainVert = 0.15                # 0.2

# patch 내 거리 분산을 고려??
noiseVariance = 0.2

# 1x7 배열, 좌우대칭, 사용 의미는??
matchedFilterKernel = [17,20,25,26,25,20,17]
obstacleArrayHorz = np.zeros((7))             # 수평방향에 장애물 확인

# ??
matchedFilterExpectedResult = 3304
matchedFilterMargin = 800


def displayObstacleArrayHorz(obstacleMap) :
    # obstacleMap 변수에 따라 수정 가능성
	displayImage = np.ones([200,obstacleMap.shape[1]*6,3]) 	# [200,obstacleMap's colx6]
	displayImage = displayImage * 255
	# Input font in image
	fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
	fontScale = 0.5
	thickness = 2

	# Opencv에서 Point라는 함수 OR Point라는 object
	testPos_x = 10
	testPos_y = 70
	testPos = (testPos_x, testPos_y)		# 10x70 xy좌표계를 생성?
	indexObstacle = 0
	
	# obstacleMap 변수에 따라 수정 가능성
	for i in range(0,obstacleMap.shape[1],10): 
		depthThis = math.floor(obstacleMap[0,i])
        
		# obstacleArrayHorz 선언 되는 부분이 어디??, 헤더파일에 위치
		obstacleArrayHorz[indexObstacle] = depthThis
		depthString = str(depthThis)
        
		# putText 함수 : 이미지내에 글씨 쓰기 
		cv2.putText(displayImage, depthString, testPos, fontFace, fontScale, color=(255,0,0), thickness=thickness)

		# testPos 업데이트 (testPos += Point(50,0))
		testPos_x += 50
		testPos = (testPos_x, testPos_y)
		indexObstacle += 1
    
	cv2.imshow("view2", displayImage)
	cv2.waitKey(1)


# 기존 코드는 전체 이미지를 기준으로 분석했지만 
# 실험 환경을 고려해 일부 이미지만을 활용
def displayCollision(depthImage):
	half_height = 60
	half_width  = 160
	CollisionImage = depthImage[half_height-50:half_height+50, half_width-100:half_width+100]

	minVal = cv2.minMaxLoc(CollisionImage)[0]
	warningVal = 0.7
	# print(minVal)
	# 기존 >> minVal <= 0.5
	if(minVal <= warningVal):				
		print("Warning : Collision")
		return 0
	elif (minVal <= 5):
		return math.sqrt((minVal-warningVal)/(4.3))+0.1
	else:
		return 1


# Corner를 인지하고 회피 Gain 업데이트
# 어떤 상황을 말하는 건지 확실히 알기
def detectCorner(depthImage):
	global lambdaObstacleHorz
	global obstacleDistanceGainHorz
	matchedFilterResult = np.dot((matchedFilterKernel), obstacleArrayHorz)		# (7,1) x (1,7)
	
	if (abs(matchedFilterResult - matchedFilterExpectedResult) < matchedFilterMargin ):
		lambdaObstacleHorz = lambdaObstacleHorzAggressive;
		obstacleDistanceGainHorz = obstacleDistanceGainHorzAggressive
		# print("Warning : Detect Corner & Update Gain")
	else:
		lambdaObstacleHorz = lambdaObstacleHorzNormal;
		obstacleDistanceGainHorz = obstacleDistanceGainHorzNormal


'''
 * Returns the angular velocity outputted by the behaviour arbitration scheme
 * This is the obstacle avoid behaviour, should be summed with heading goal
'''
@njit(fastmath=True)
def avoidObstacle(seg_center) :
    image_height = 270
    image_width  = 640
    # seg_center = np.array([[100, 135, 5, 5, 100],[100, 135, 5, 50, 100]])
    # seg_center = np.array([[540, 135, 5, 5, 100]])
    L = seg_center.shape[0]
    
    image_size = image_height*image_width
    seg_avg_size = image_size / L
    
    # detectCorner(depthImage)
    # velocity = displayCollision(depthImage)

    Horz_fObstacleTotal = 0
    Vert_fObstacleTotal = 0
	
    for i in range(L):
        if seg_center[i,3]>20:
            seg_center[i,3] = 20
        
        # print(seg_center[i])
        # View Angle (-FOV/2 < x < +FOV/2)
        Horz_obstacleBearing = (hfov*seg_center[i,0]/image_width) - (hfov/2) + (hfov/2/image_width)
        Vert_obstacleBearing = (vfov*seg_center[i,1]/image_height) - (vfov/2) + (vfov/2/image_height)
        
        # print(Horz_obstacleBearing)
        
        # degree to radian
        Horz_obstacleBearing = Horz_obstacleBearing * math.pi / 180
        Vert_obstacleBearing = Vert_obstacleBearing * math.pi / 180
        
        # The larger the center, the smaller the border.
        Horz_bearingExponent = np.exp(-1*pow(Horz_obstacleBearing,2)/(2*pow(angularRangeHorz,2)))
        Vert_bearingExponent = np.exp(-1*pow(Vert_obstacleBearing,2)/(2*pow(angularRangeVert,2)))

        distanceExponent = np.exp(-obstacleDistanceGainHorz * seg_center[i,3]) 

        Horz_fObstacleTotal += Horz_obstacleBearing * Horz_bearingExponent * distanceExponent * seg_center[i,4]
        Vert_fObstacleTotal += Vert_obstacleBearing * Vert_bearingExponent * distanceExponent
        
        # print(Horz_obstacleBearing * Horz_bearingExponent * distanceExponent * (image_width/L)
        
    # 단위는 degree
    yaw = Horz_fObstacleTotal * lambdaObstacleHorz / 275
    Thrust = Vert_fObstacleTotal * lambdaObstacleVert
    
    # print(yaw)
    
    return yaw, Thrust

def followGoal(goalAngle, currentBearing) :
    return -lambdaGoalHorz * math.sin(currentBearing - goalAngle)

def sumBehavioursHorz(angVelAvoidHorz, angVelFollowGoal):
    behaviourSum = weightObstacleHorz * angVelAvoidHorz + weightGoalHorz*angVelFollowGoal
    return behaviourSum


# def display_navi(Base_img, vel1, vel2, vel3):
    	
#     HFOV = 117
#     vel_max = 3 # [m/s]

#     img = Base_img 
#     img_H,img_W,__ = img.shape

#     margin_W = 150
#     margin_H = 50

#     # Window init
#     window = np.ones((img_H+2*margin_H,img_W+2*margin_W,3), dtype=np.uint8)*255
#     window[margin_H:margin_H+img_H, margin_W:margin_W+img_W] = img[:,:,:]
#     window_H,window_W,__ = window.shape

#     cv2.imshow("test1",window)

#     # Yaw UI
#     yaw_pivot_w = 3
#     yaw_pivot_h = 30
#     yaw_pivot_c = int(window_H//1.5)
#     yaw_stick = int(img_W//2 * vel3 / 180)
#     if vel3>0:
#         window[yaw_pivot_c-yaw_pivot_h:yaw_pivot_c+yaw_pivot_h, window_W//2:window_W//2+yaw_stick] = [0,0,255]
#     else:
#         window[yaw_pivot_c-yaw_pivot_h:yaw_pivot_c+yaw_pivot_h, window_W//2+yaw_stick:window_W//2] = [0,0,255]
#     window[yaw_pivot_c-yaw_pivot_h:yaw_pivot_c+yaw_pivot_h, window_W//2-yaw_pivot_w:window_W//2+yaw_pivot_w] = [255,0,0]

#     # Vel1 UI
#     vel1_graph_w = 50
#     vel1_graph_h = img_H//2
#     vel1_graph_pos_w = margin_W//2
#     vel1_graph_pos_h = window_H//2
#     vel1_graph_stick = int(img_H*vel1/3)
#     window[img_H+margin_H-vel1_graph_stick:img_H+margin_H,vel1_graph_pos_w-vel1_graph_w:vel1_graph_pos_w+vel1_graph_w] = [0,255,0]
#     window = cv2.rectangle(window,(vel1_graph_pos_w-vel1_graph_w, vel1_graph_pos_h-vel1_graph_h), (vel1_graph_pos_w+vel1_graph_w, vel1_graph_pos_h+vel1_graph_h), (0,0,0), 2)

#     # Vel2 UI
#     vel1_graph_w = 50
#     vel1_graph_h = img_H//2
#     vel1_graph_pos_w = margin_W + img_W + margin_W//2
#     vel1_graph_pos_h = window_H//2
#     vel1_graph_stick = int(img_H*vel2/3)
#     if vel2>0:
#         window[vel1_graph_pos_h:vel1_graph_pos_h+vel1_graph_stick,vel1_graph_pos_w-vel1_graph_w:vel1_graph_pos_w+vel1_graph_w] = [0,255,0]
#     else:
#         window[vel1_graph_pos_h+vel1_graph_stick:vel1_graph_pos_h,vel1_graph_pos_w-vel1_graph_w:vel1_graph_pos_w+vel1_graph_w] = [0,255,0]

#     window = cv2.rectangle(window,(vel1_graph_pos_w-vel1_graph_w, vel1_graph_pos_h-vel1_graph_h), (vel1_graph_pos_w+vel1_graph_w, vel1_graph_pos_h+vel1_graph_h), (0,0,0), 2)

#     cv2.imshow('Navigation',window)
#     cv2.waitKey(0)
	
#     print(vel1, vel2, vel3)