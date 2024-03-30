import cv2
import numpy as np
import math

def getMid(frame):
    #print("height:",frame.shape[0])
    #print("width:",frame.shape[1])
    #return mid_height,mid_width
    return frame.shape[0]//2,frame.shape[1]//2

def subtractFarPixels(frame, depth, threshold):
    for i in range(0,depth.shape[0]):
        for j in range(0, depth.shape[1]):
            if(depth[i][j]<threshold):
                frame[i][j]=[255,255,255]
    return frame
def depth_to_distance(depth_value,depth_scale=1.0):
    return 1.0/((depth_value*depth_scale)+0.000000001)

def to_pixel(dim):
    return math.ceil(dim*37.795)
def getMatrixAvg(matrix):
    h = matrix.shape[0]
    w = matrix.shape[1]
    top_left = np.array([matrix[i][:w//2] for i in range(h//2)])
    top_right = np.array([matrix[i][w//2:] for i in range(h//2)])
    bot_left = np.array([matrix[i][:w//2] for i in range(h//2, h)])
    bot_right = np.array([matrix[i][w//2:] for i in range(h//2, h)])

    #topleft, top right, bot right, bot left
    return np.mean(top_left.flatten()),np.mean(top_right.flatten()),np.mean(bot_right.flatten()),np.mean(bot_left.flatten())

def vote(optDir,depthDir):
    if(optDir!=depthDir[0]):
        return depthDir[0]
    return optDir

def getFlowDir(leftTop,leftBot,rightTop,rightBot):
    min = 900
    dir = ""
    if (min > leftTop):
        min = leftTop
        dir = "Left Top"
    if (min > leftBot):
        min = leftBot
        dir = "Left Bottom"
    if (min > rightTop):
        min = rightTop
        dir = "Right Top"
    if (min > rightBot):
        min = rightBot
        dir = "Right Bottom"
    return dir

def getDepthDir(tl,bl,tr,br,threshold):
    min = 1000
    dir = {tl:"Left Top",bl:"Left Bottom",tr:"Right Top",br:"Right Bottom"}
    if(tl>threshold):
        del dir[tl]
    if(bl>threshold):
        del dir[bl]
    if(tr>threshold):
        del dir[tr]
    if(br>threshold):
        del dir[br]
    if(len(dir)==0):
        #if empty
        dir[0]="STOP"
    dict(sorted(dir.items()))
    return list(dir.values())
def findNorms(output_norm,frame,results):
    # randomly sample 3 points from a certain region
    # fit a plane to them and calculate the norm of the plane
    startX = frame.shape[1]/9*2
    startY = frame.shape[0]/9*6
    endX = frame.shape[1]/9*7
    endY = frame.shape[0]
    best_plane = None
    most_counts = 0
    for _ in range(2):
        population = 700
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
                else:
                    continue
                if len(tmp) == 3:
                    # check non-colinear
                    AB = np.array(tmp[0]) - np.array(tmp[1])
                    BC = np.array(tmp[1]) - np.array(tmp[2])
                    cross_prod = np.cross(BC, AB)
                    if (tmp[0][0]-tmp[1][0] == 0 and tmp[0][0]-tmp[2][0]==0) or (tmp[0][1] - tmp[1][1] == 0 and tmp[0][1] - tmp[2][1] == 0):
                        # colinear
                        tmp.clear()
                    else:
                        d = cross_prod[0] * X + cross_prod[1] * Y + cross_prod[2] * Z
                        norms.append([cross_prod, d])
            if tmp not in vectors_sets:
                vectors_sets.append(tmp)
            else:
                norms.remove(len(norms) - 1)
        # find the norm from norms that fit most points from a rectangular region (startX,startY) to (endX,endY)
        # Note Z is the value at output_norm[Y][X]

        if best_plane != None:
            norms.append(best_plane)
        counts = [0] * len(norms)
        if most_counts > 0:
            counts[len(counts) - 1] = most_counts
        tested = []
        while len(tested) < 300:
            X = np.random.randint(startX, endX)
            Y = np.random.randint(startY, endY)
            Z = output_norm[Y][X]
            if [X, Y, Z] not in tested:
                tested.append([X, Y, Z])
                for i in range(len(norms)):
                    fit = norms[i][0][0] * X + norms[i][0][1] * Y + norms[i][0][2] * Z
                    if abs(abs(fit) - abs(norms[i][1])) <= 50.0:
                        counts[i] += 1
        best_fit_idx = np.argmax(counts)
        best_plane = norms[best_fit_idx]
        most_counts = counts[best_fit_idx]
        results[0] = best_plane
