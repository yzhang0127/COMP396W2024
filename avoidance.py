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



    last_center = [4,4]
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
            t1 = threading.Thread(target=utils.findNorms,args=(output_norm,mid_height,mid_width,frame,results))
            #best_plane = utils.findNorms(output_norm,mid_height,mid_width,frame)
            t1.start()

            #find a region that has least amount of objects
            safe_region = [[0]]
            safe_tiles = []
            visited_submat = []
            visited_centers = []
            found = False
            c_x = 0
            c_y = 0
            first_try = True
            while len(visited_centers)<=8 and not found:
                #choose a random center
                #x,y are the coordinates of the left top cornor of the tile
                if first_try:
                    y = last_center[1]
                    x = last_center[0]
                    first_try = False
                else:
                    y = random.randint(4,6)
                    x = random.randint(4,6)
                    while [x,y] in visited_centers:
                        y = random.randint(4, 6)
                        x = random.randint(4, 6)
                visited_centers.append([x,y])
                safe_region.append([])
                #start to span a rectangle
                c_x = x
                c_y = y
                queue = [[x,y]]
                dis_to_center = 1
                #rectangle will be large enough when the first coord that is 3 tiles away from the center is popped
                while len(queue)>0:
                    coord = queue.pop(0)
                    x = coord[0]
                    y = coord[1]

                    if abs(x-c_x)>=dis_to_center or abs(y-c_y)>=dis_to_center:
                        #safe region is a 3 * 3 square
                        #the distance from the edge of the rectangle to the center is 1 tile
                        found = True
                        last_center=[c_x,c_y]
                        break


                    if coord in visited_submat and coord not in safe_tiles:
                        # unable to span a rectangle
                        break
                    elif coord in visited_submat and coord in safe_tiles:
                        safe_region[len(safe_region) - 1].append([x,y])
                    elif coord not in visited_submat:
                        #not visited
                        visited_submat.append(coord)
                        subMat = output_norm[y * height_interval:height_interval + y * height_interval,
                                    x * width_interval:width_interval + x * width_interval]

                        subMat = [[0.01 if a < 0.4 else a for a in row] for row in subMat]

                        if y >= 4:
                            t1.join()
                            best_plane = results[0]
                            print(best_plane)
                            for i in range(0, len(subMat)):
                                for j in range(0 ,len(subMat) ):
                                    d = best_plane[0][0] * j + best_plane[0][1] * i + best_plane[0][2] * output_norm[i+y*height_interval][j+x*width_interval]
                                    if abs(abs(d) - abs(best_plane[1])) < 39.0:
                                        subMat[i][j] = 0.01


                        if np.mean(subMat)<=0.02 and y<=4:
                            #safe
                            safe_region[len(safe_region)-1].append([x,y])
                            safe_tiles.append([x,y])
                        else:
                            #not safe break
                            break

                    #expand surronding tiles
                    tmp_list = []
                    if (y+1) < 9:
                        if [x,y+1] not in queue:
                            tmp_list+=[[x,y+1]]
                        if (x+1) < 9:
                            if [x+1,y+1] not in queue:
                                tmp_list+=[[x+1,y+1]]
                            if [x+1,y] not in queue:
                                tmp_list+=[[x+1,y]]
                        if (x-1) >= 0:
                            if [x - 1, y + 1] not in queue:
                                tmp_list += [[x - 1, y + 1]]
                            if [x - 1, y] not in queue:
                                tmp_list += [[x - 1, y]]
                    if (y-1)>=0:
                        if [x,y-1] not in queue:
                            tmp_list+=[[x,y-1]]
                        if (x+1) < 9:
                            if [x+1,y-1] not in queue:
                                tmp_list+=[[x+1,y-1]]

                        if (x-1) >= 0:
                            if [x - 1, y - 1] not in queue:
                                tmp_list += [[x - 1, y - 1]]
                    queue+=tmp_list
            #find the largest one
            max_len = 0
            max_region = []
            safe_region[0].pop(0)
            for region in safe_region:
                if len(region)>max_len and len(region)>=25:
                    max_len = len(region)
                    max_region = region

            if not found:
                leftSubMat = output_norm[0:9*height_interval,
                                    0:mid_width]

                rightSubMat = output_norm[0:9*height_interval,
                                    mid_width:9*width_interval]

                leftMean = np.mean(leftSubMat)
                rightMean = np.mean(rightSubMat)
                print([leftMean,rightMean])
                if leftMean < rightMean:
                    if leftMean <= 0.45:
                        cv2.rectangle(frame, (0, 0), (3*width_interval, 9*height_interval), (0, 255, 255), 2)
                    else:
                        print("stop")
                else:
                    if rightMean <= 0.45:
                        cv2.rectangle(frame, (5*width_interval, 0), (9 * width_interval, 9 * height_interval), (0, 255, 255), 2)
                    else:
                        print("Stop")
            else:

                sX = (c_x-(dis_to_center+1))*width_interval
                sY = (c_y-(dis_to_center+1))*height_interval
                eX = (c_x+(dis_to_center+1))*width_interval+width_interval
                eY = (c_y+(dis_to_center+1))*height_interval+height_interval
                cv2.rectangle(frame,(sX,sY),(eX,eY),(255,0,0),2)
                #cv2.rectangle(frame,(4*width_interval,4*height_interval),(4*width_interval+width_interval,4*height_interval+height_interval),(255,255,0),2)#draw center
                cv2.line(frame,(0,9*height_interval),(sX,eY),(255,0,0),2)
                cv2.line(frame, (9*width_interval, 9 * height_interval), (eX, eY), (255, 0, 0), 2)
            '''
            #filter ground
            #for demo only
            for i in range (0,output_norm.shape[0]):
                for j in range(0,output_norm.shape[1]):
                    d = best_plane[0][0]*j+best_plane[0][1]*i+best_plane[0][2]*output_norm[i][j]
                    if abs(abs(d)-abs(best_plane[1])) < 39.0:
                        output_norm[i][j]=0.01
            '''








        #update interval
        frame_idx += 1

        #calculate frame rate
        end = time.time()
        diff_time = end-start #unit is sec/frame
        frame_rate = round(1/diff_time,2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(depth_out,"FPS "+str(frame_rate),(50,50),font,1,(255,255,255),2,cv2.LINE_4)
        cv2.imshow("depth image", depth_out)
        cv2.imshow("processed",output_norm)
        cv2.imshow("Original",frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



estimation(10,5.2,"MiDaS_small",1)