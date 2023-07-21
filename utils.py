import cv2
import numpy as np

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
