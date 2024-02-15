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
        cv2.imshow("Original",frame)
        #get depth info
        depth = getDepth(transform,frame,midas)
        np.fmod(depth,360)
        output_norm = cv2.normalize(depth,None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #Note: 0<output_norm[i][j]<1
        #      larger the number is, closer to the object
        # find a submatrix with lowest average of depth
        mid_height,mid_width = utils.getMid(output_norm)

        '''#filter out objects that is far away
        filter_func = numpy.vectorize(filter)
        result = filter_func(output_norm)
        cv2.imshow("filtered depth", result)'''
        depth_out = copy.deepcopy(output_norm)
        #cut frame into 9*9 subframes evenly
        width_interval = math.floor(frame.shape[1]/9)
        height_interval = math.floor(frame.shape[0]/9)
        mat = [[0 for _ in range(9)] for _ in range(9)]

        if mode == "ground":
            #A groud based robot can't move upwards

            startX = mid_width - math.ceil(mid_width / 4)
            startY = mid_height
            endX = mid_width + math.ceil(mid_width / 4)
            endY = frame.shape[0] - 1


            #randomly sample 3 points from a certain region
            #fit a plane to them and calculate the norm of the plane
            if frame_idx==1:
                print("calibrating finished")
            best_plane = None
            if frame_idx<=1:
                print("calibrating ground plane, please don't move")
                most_counts = 0
                for _ in range(10):
                    population = 500
                    vectors_sets = []
                    norms = []
                    while len(vectors_sets) < population:
                        tmp = []
                        while len(tmp) < 3:
                            X = np.random.randint(startX, endX)
                            Y = np.random.randint(startY, endY)
                            Z = output_norm[Y][X]
                            if [X, Y, Z] not in tmp:
                                tmp.append([X, Y, Z])
                            if len(tmp) == 3:
                                # check non-colinear
                                AB = np.array(tmp[0]) - np.array(tmp[1])
                                BC = np.array(tmp[1]) - np.array(tmp[2])

                                cross_prod = np.cross(BC, AB)
                                norm = np.linalg.norm(cross_prod)
                                if norm < 10.0:
                                    # colinear
                                    tmp.clear()
                                else:
                                    d=cross_prod[0]*X+cross_prod[1]*Y+cross_prod[2]*Z
                                    norms.append([cross_prod,d])
                        if tmp not in vectors_sets:
                            vectors_sets.append(tmp)
                        else:
                            norms.remove(len(norms)-1)
                    # find the norm from norms that fit most points from a rectangular region (startX,startY) to (endX,endY)
                    #Note Z is the value at output_norm[Y][X]

                    if best_plane != None:
                        norms.append(best_plane)
                    counts = [0] * len(norms)
                    if most_counts>0:
                        counts[len(counts)-1] = most_counts
                    tested = []
                    while len(tested)<population*1.4:
                        X = np.random.randint(startX, endX)
                        Y = np.random.randint(startY, endY)
                        Z = output_norm[Y][X]
                        if [X,Y,Z] not in tested:
                            tested.append([X,Y,Z])
                            for i in range(len(norms)):
                                fit = norms[i][0][0]*X+norms[i][0][1]*Y+norms[i][0][2]*Z
                                if abs(abs(fit)-abs(norms[i][1]))<=50.0:
                                    counts[i]+=1
                    best_fit_idx = np.argmax(counts)
                    best_plane = norms[best_fit_idx]
                    most_counts = counts[best_fit_idx]

            #find a region that has least amount of objects
            safe_region = [[0]]
            safe_tiles = []
            visited_submat = []
            visited_centers = []

            while len(visited_centers)<=24:
                #choose a random center
                #x,y are the coordinates of the left top cornor of the tile
                y = random.randint(2,6)
                x = random.randint(2,6)
                while [x,y] in visited_centers:
                    y = random.randint(2, 6)
                    x = random.randint(2, 6)
                visited_centers.append([x,y])
                safe_region.append([])
                #start to span a rectangle
                c_x = x
                c_y = y
                queue = [[x,y]]
                #rectangle will be large enough when the first coord that is 3 tiles away from the center is popped
                while len(queue)>0:
                    coord = queue.pop(0)
                    x = coord[0]
                    y = coord[1]
                    if abs(x-c_x)>=3 or abs(y-c_y)>=3:
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
                        subMat = [[0.01 if a <= 0.28 else a for a in row] for row in subMat]
                        if i >= mid_height:
                            for i in range(0, len(subMat)):
                                for j in range(0 ,len(subMat) ):
                                    d = best_plane[0][0] * j + best_plane[0][1] * i + best_plane[0][2] * output_norm[i+y*height_interval][j+x*width_interval]
                                    if abs(abs(d) - abs(best_plane[1])) <= 30:
                                        subMat[i][j] = 0.01

                        if np.mean(subMat)<=0.02:
                            #safe
                            safe_region[len(safe_region)-1].append([x,y])
                            safe_tiles.append([x,y])
                        else:
                            #not safe break
                            break

                    #expand surronding tiles
                    tmp_list = []
                    if (y+1) < output_norm.shape[0]:
                        if [x,y+1] not in queue:
                            tmp_list+=[[x,y+1]]
                        if (x+1) < output_norm.shape[1]:
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
                        if (x+1) < output_norm.shape[1]:
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
            if len(max_region)==0:
                print("Stop")

            for coord in max_region:
                sX = coord[1]*width_interval
                sY = coord[0]*height_interval
                eX = sX+width_interval
                eY = sY+height_interval
                cv2.rectangle(output_norm,(sX,sY),(eX,eY),(203,192,255),thickness=2)












            '''
            #filter ground
            #for demo only
            for i in range (0,output_norm.shape[0]):
                for j in range(0,output_norm.shape[1]):
                    d = best_plane[0][0]*j+best_plane[0][1]*i+best_plane[0][2]*output_norm[i][j]
                    if abs(abs(d)-abs(best_plane[1])) <= 30.0:
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

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



estimation(10,5.2,"MiDaS_small",1)