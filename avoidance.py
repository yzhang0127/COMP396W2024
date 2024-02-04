import copy
import sys

import cv2
import numpy
import torch
import time
import numpy as np
import random
import math
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import  StandardScaler
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
        if mode == "ground":
            #A groud based robot can't move upwards

            startX = mid_width - math.ceil(mid_width / 4)
            startY = mid_height
            endX = mid_width + math.ceil(mid_width / 4)
            endY = frame.shape[0] - 1


            #randomly sample 3 points from a certain region
            #fit a plane to them and calculate the norm of the plane
            population=100
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
            counts = [0]*len(norms)
            tested = []
            while len(tested)<100:
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







            #filter ground

            #for demo only
            for i in range (mid_height,output_norm.shape[0]):
                for j in range(0,output_norm.shape[1]):
                    d = best_plane[0][0]*j+best_plane[0][1]*i+best_plane[0][2]*output_norm[i][j]
                    if abs(abs(d)-abs(best_plane[1])) <= 30.0:
                        output_norm[i][j]=0.01






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

def filter(x):
    if x<0.09:
        return 0.00001
    return x

estimation(10,5.2,"MiDaS_small",1)