import cv2
import torch
import time
import numpy as np

import utils
from MiDasDepth import getDepth


#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()



midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 20,
                    qualityLevel = 0.3,
                    minDistance = 10,
                    blockSize = 7 )


trajectory_len = 40
detect_interval = 5
trajectories = []
frame_idx = 0

cap = cv2.VideoCapture(0)
while cap.isOpened():

    start = time.time()

    ret,frame = cap.read()
    cv2.imshow("Original",frame)

    #get depth info
    depth = getDepth(transform,frame,midas)

    # get average depths in four quadrants in the depth map
    tl, tr, br, bl = utils.getMatrixAvg(depth)
    #subtract all pixels that is far away
    threshold = 560
    #cv2.imshow("subtraction",frame2)


    #get optical flow
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()
    mid_y, mid_x = utils.getMid(frame)
    # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []
        leftTop, leftBot, rightTop, rightBot = 0, 0, 0, 0
        # Get all the trajectories
        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue

            #screan out good points that is far away
            i = int(x)-1
            j = int(y)-1
            if i >= depth.shape[1]:
                i = depth.shape[1]-1
            if j >= depth.shape[0]:
                j = depth.shape[0]-1
            if(depth[j][i]<threshold):
                trajectory.append((x, y))


            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            # Newest detected point
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)


            if (int(x) < mid_x and int(y) < mid_y):
                leftTop += 1
            elif (int(x) < mid_x and int(y) > mid_y):
                leftBot += 1
            elif (int(x) > mid_x and int(y) > mid_y):
                rightBot += 1
            elif(int(x) > mid_x and int(y) < mid_y):
                rightTop += 1
        trajectories = new_trajectories

        #find the direction with minimum objects
        min = 900
        dir = "STOP"
        #print("LT",leftTop,"LB",leftBot,"RT",rightTop,"RB",rightBot)

        # safety check
        depth_avg = np.mean(depth.flatten())
        if (depth_avg >= 800):
            # running into walls
            print("STOP")
            break


        if (depth_avg <= 370):
            print("Forward")
        else:
            optDir = utils.getFlowDir(leftTop, leftBot, rightTop, rightBot)
            depthDir = utils.getDepthDir(tl, bl, tr, br, threshold)
            dir = utils.vote(optDir, depthDir)
            print("Go",dir)

        # Draw all the trajectories
        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Lastest point in latest trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            # If good features can be tracked - add that to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end - start)

    # Show Results
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Optical Flow', img)
    cv2.imshow('Mask', mask)

    #update interval

    frame_idx += 1



    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()