# COMP396_ObstcleAvoidance

## Name
Comp 396 Project - Robot Vision  
Project: Obstacle Avoidance Robot Motion  
Member: Yijun Zhang  


## Description
The aim of this project is to facilitate navigation and obstacle avoidance for autonomous robots while minimizing the expenditure on peripherals. This project involves using python, pytorch and openCV to process an input image from a RGB camera to find a safe path for the autonomous robot to follow.

## Visuals
For all the specific test cases or demo, please read [https://docs.google.com/document/d/1FO6Ub0ffqF-ezLOUPB6URQQ-g_7779S6S4RxHKvHr-Q/edit?usp=sharing](https://docs.google.com/document/d/1WRCtEa9kzLKDQuPQ54piGrdv7e0YX7yQ/edit?usp=sharing&ouid=105139983514988380865&rtpof=true&sd=true)


## Installation
1. Clone the files into your local device and create a project of it using Python 3.8
2. Either using ``` pip -install ``` or your IDE to install necessary libraries in the ``` requirements.txt ```
3. Connect to the internet and run the avoidance.py  
    This step will automatically install the neural network into your ``` ./cache ``` directory. It may take a while to download but once it is successfully installed, you no longer need to connect to the internet.  
4. To terminate the program, you can press key "q" or click the termination button in your IDE.  
NOTE: This program takes some time to load the network and connect to the camera. Please wait until the video is shown on screen.  
If your device has a built in camera or your device has no other video transmission card nor extra cameras, please call the function, ``` def estimation(dimX, dimY, modelName,cam_channel=0,mode="ground") ```, with ``` estimation(10,5.2,"MiDaS_small",0) ``` to run the avoidance.py. If you want to use the camera besides the one already in your device, you can try to set the ``` cam_channel ``` to 1 or 2 depends on the number of video devices connected.

