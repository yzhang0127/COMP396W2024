import copy
import sys

import cv2
import numpy
import torch
import time
import numpy as np
import random
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



            population = 2000
            if frame_idx%25==0:
                plane = []
                # Generate random indices without replacement
                indices = np.random.choice(output_norm.shape[1], size=(population, 3), replace=True)
                y_indices = np.random.randint(mid_height, output_norm.shape[0], size=(population, 3))

                # Create array with random points
                points = np.stack([indices, y_indices, output_norm[y_indices, indices]], axis=-1)

                # Calculate vectors and cross products
                AB = points[:, 1, :] - points[:, 0, :]
                AC = points[:, 2, :] - points[:, 0, :]
                norms = np.cross(AB, AC)
                # Extract X, Y, Z components
                X=[norm[0] for norm in norms[:population]]
                Y=[norm[1] for norm in norms[:population]]
                Z=[norm[2] for norm in norms[:population]]

                # Create a RANSAC regressor
                regressor = make_pipeline(StandardScaler(),RANSACRegressor())
                # Reshape X, Y, Z to column vectors
                X = np.array(X).reshape(-1,1)
                Y = np.array(Y).reshape(-1,1)
                Z = np.array(Z).reshape(-1,1)
                #fit the regressor
                regressor.fit(np.hstack([X,Y]),Z)
                # Retrieve the coefficients of the plane
                a,b= regressor.named_steps['ransacregressor'].estimator_.coef_[0]
                c = regressor.named_steps['ransacregressor'].estimator_.intercept_
            plane = [a,b,c]
            #filter ground
            threshold = 0.23
            scale1 = 1e8

            '''
            #for demo only
            for i in range (mid_height,output_norm.shape[0]):
                for j in range(0,output_norm.shape[1]):

                    value1 = abs(a*j+b*i+c)/scale1
                    if np.abs(value1-output_norm[i][j])<=threshold :
                        output_norm[i][j]=0.001
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

def filter(x):
    if x<0.09:
        return 0.00001
    return x

estimation(10,5.2,"MiDaS_small",1)