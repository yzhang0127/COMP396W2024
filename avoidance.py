import copy
import sys

import cv2
import numpy
import torch
import time
import numpy as np
import random
import math
import itertools
import matplotlib.pyplot as plt

import utils
import threading
from MiDasDepth import getDepth

def load_model(model_type):
    #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()



    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return transform,midas

def estimation(dimX, dimY, modelName,cam_channel=0,mode="ground"):

    '''

    :param dimX: width of the robot in cm
    :param dimY: height of the robot in cm
    :param modelName: "DPT_Large", "DPT_Hybrid","MiDaS_small"
    :param mode: "ground" or "sky"
    :return:
    '''

    depth_scale = 1.0
    frame_idx = 0
    height = utils.to_pixel(dimY)
    width = utils.to_pixel(dimX)
    transform,midas = load_model(modelName)

    #opencv has default cam input and output : 640x480
    #default cam stream channel is 0
    cap = cv2.VideoCapture(cam_channel,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)



    while cap.isOpened():

        start = time.time()
        ret,frame = cap.read()

        #get depth info
        depth = getDepth(transform,frame,midas)
        np.fmod(depth,360)
        output_norm = cv2.normalize(depth,None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #Note: 0<output_norm[i][j]<1
        #      larger the number is, closer to the object
        # find a submatrix with lowest average of depth
        mid_height,mid_width = utils.getMid(output_norm)


        depth_out = copy.deepcopy(output_norm)
        #cut frame into 9*9 subframes evenly
        width_interval = math.floor(frame.shape[1]/9)
        height_interval = math.floor(frame.shape[0]/9)
        mat = [[0 for _ in range(9)] for _ in range(9)]


        if mode == "ground":
            #A groud based robot can't move upwards

            results = [None]*1
            t1 = threading.Thread(target=utils.findNorms,args=(output_norm,frame,results))
            #best_plane = utils.findNorms(output_norm,mid_height,mid_width,frame)
            t1.start()

            safe_region_size = 4
            visited_safe = []
            visited_unsafe = []
            tried_center = []
            safe_center_count = []
            best_center = []
            candidate_center = None
            min_rand = 4
            max_rand = 8 - safe_region_size + 1

            while len(tried_center)< pow((max_rand-min_rand+1),2):

                queue = []
                visited = []
                c_x = 4
                c_y = 4
                if len(tried_center)>=0:
                    while [c_x,c_y] in tried_center:
                        c_x = random.randint(2,max_rand)
                        c_y = random.randint(min_rand,max_rand)

                queue.append([c_x,c_y])
                tried_center.append([c_x,c_y])

                safe_count = 0
                while len(queue)>0:
                    coord = queue.pop(0)
                    visited.append(coord)
                    x = coord[0]
                    y = coord[1]
                    if coord in visited_unsafe:
                        break
                    elif coord not in visited_safe:
                        subMat = output_norm[y * height_interval:height_interval + y * height_interval,
                                 x * width_interval:width_interval + x * width_interval]

                        subMat = [[0.01 if a <= 0.5 else a for a in row] for row in subMat]
                        # filter out ground
                        if y >= 5:
                            t1.join()
                            best_plane = results[0]

                            for i in range(0, len(subMat)):
                                for j in range(0, len(subMat)):
                                    r_x = j + x * width_interval
                                    r_y = i + y * height_interval
                                    d = best_plane[0][0] * r_x + best_plane[0][1] * r_y + best_plane[0][2] * \
                                        output_norm[r_y][r_x]
                                    if abs(abs(d) - abs(best_plane[1])) <= 20.0:
                                        subMat[i][j] = 0.01
                            if np.mean(subMat) < 0.6:
                                # safe
                                visited_safe.append(coord)
                                safe_count+=1
                            else:
                                visited_unsafe.append(coord)
                                break
                        else:
                            # print("ground: ",np.mean(subMat))
                            if np.mean(subMat) <= 0.02:
                                # safe
                                visited_safe.append(coord)
                                safe_count+=1
                            else:
                                visited_unsafe.append(coord)
                                break
                    else:
                        safe_count+=1

                    #expand
                    if x-1>=0 and x+1<9 and y-1>=0 and y+1<9:
                        tmp = [[x,y+1],[x,y-1],[x+1,y],[x-1,y]]
                        for c in tmp:
                            if c not in visited and c not in queue:
                                if abs(c[0]-c_x)<=safe_region_size-1 and abs(c[1]-c_y)<=safe_region_size-1:
                                    queue.append(c)

                safe_center_count.append(safe_count)
                if safe_count==35:
                    best_center = [c_x,c_y]
                    break
            if best_center == []:
                #unable to find a large enough region
                max_count = 0
                idx = None
                for i in range(0,len(safe_center_count)):
                    if safe_center_count[i]>=33 and safe_center_count[i]>max_count:
                        idx = i
                if idx != None:
                    candidate_center = tried_center[i]
            else:
                candidate_center = best_center

            if candidate_center!=None:
                sX = (candidate_center[0]-safe_region_size+1)*width_interval
                sY = (candidate_center[1]-safe_region_size+1)*height_interval
                eX = (candidate_center[0]+safe_region_size)*width_interval
                eY = (candidate_center[1]+safe_region_size)*height_interval
                cv2.rectangle(frame,(sX,sY),(eX,eY),(255,0,0),2)
            else:
                left = output_norm[0:frame.shape[1],0:mid_width]
                right = output_norm[0:frame.shape[1],mid_width:frame.shape[0]]
                meanL = np.mean(left)
                meanR = np.mean(right)
                if meanR>=0.6 and meanL>=0.6:
                    print("STOP")
                else:
                    if meanL - meanR >= 0.1:
                        cv2.rectangle(frame,(mid_width,0),(frame.shape[1],frame.shape[0]),(0,255,255),2)
                    else:
                        cv2.rectangle(frame, (0, 0), (mid_width, frame.shape[0]),(0,255,255),2)


            '''
            #filter ground
            #for demo only
            t1.join()
            best_plane = results[0]
            for i in range (mid_height,output_norm.shape[0]):
                for j in range(0,output_norm.shape[1]):
                    d = best_plane[0][0]*j+best_plane[0][1]*i+best_plane[0][2]*output_norm[i][j]
                    if abs(abs(d)-abs(best_plane[1])) < 40.0:
                        output_norm[i][j]=0.01
                        '''




        #update interval
        frame_idx += 1

        #calculate frame rate
        end = time.time()
        diff_time = end-start #unit is sec/frame
        frame_rate = round(1/diff_time,2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,"FPS "+str(frame_rate),(50,50),font,1,(255,255,255),2,cv2.LINE_4)
        cv2.imshow("depth image", depth_out)
        #cv2.imshow("processed",output_norm)
        cv2.imshow("Original",frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



estimation(10,5.2,"MiDaS_small",1)