import cv2
import time
import threading
import DobotDllType as dType
import numpy as np
from operator import itemgetter


CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

#Load Dll
api = dType.load()

#Connect Dobot
state = dType.ConnectDobot(api, "", 115200)[0]
print("Connect status:",CON_STR[state])

c = 'f'
cap = cv2.VideoCapture(1)
ret, frame = cap.read()
h , w  , sh = frame.shape

while(cv2.waitKey!=-1 and c!=113):
    ret, frame = cap.read()
    im = frame
    img = frame
    cv2.imshow('Pose',frame)
    c = cv2.waitKey(1)


#frame1 = cv2.imread('Po3.jpg',-1)

flag = 0
count=0
count_of_img = 0
ZZZ=[]
YYY=0
Q = []

x1=0
x2=0
y1=0
y2=0
img_size_h = h/10.0
img_size_w = w/10.0
TL_x = 300
TL_y = 80

zcord = -30.5

def water_dip():
    dType.SetPTPJumpParams(api, 25,0, isQueued = 1)
    x = 259
    y = -182
    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, x, y, 0, 0, isQueued = 1)[0]
    for i in range(0, 6):
        if i % 2 != 0:
            z = -10
        else:
            z = -40
        lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, x, y, z, 0, isQueued = 1)[0]

    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, 187.9, -205, -20, 0, isQueued = 1)[0]
    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, 250, 0, -20, 0, isQueued = 1)[0]
    
    dType.SetPTPJumpParams(api, 4,0, isQueued = 1)
    
    #Start to Execute Command Queued
    dType.SetQueuedCmdStartExec(api)

    #Wait for Executing Last Command 
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(100)
		
    #Stop to Execute Command Queued
    dType.SetQueuedCmdStopExec(api)
    
    #Clean Command Queued
    dType.SetQueuedCmdClear(api)


def nothing(x):
    pass

def fill(resized, start_coords, fill_value):
    global YYY 
    
    xsize, ysize = resized.shape
    orig_value = resized[start_coords[0], start_coords[1]]
    
    stack = set(((start_coords[0], start_coords[1]),))
    if fill_value == orig_value:
        raise ValueError("Filling region with same value "
                        "already present is unsupported. "
                        "Did you already fill this region?")
    
    while stack:
        x, y = stack.pop()
            
        if resized[x, y] == orig_value:
            resized[x, y] = fill_value
            ZZZ[YYY].append((x,y))
            if x > 0:
                stack.add((x - 1, y))
            if x < (xsize - 1):
                stack.add((x + 1, y))
            if y > 0:
                stack.add((x, y - 1))
            if y < (ysize - 1):
                stack.add((x, y + 1))
    YYY=YYY+1




cv2.namedWindow('Test Parameters_Thresholding RGB')
cv2.createTrackbar('ThresholdLH', 'Test Parameters_Thresholding RGB', 0, 255, nothing)
cv2.createTrackbar('ThresholdUH', 'Test Parameters_Thresholding RGB', 0, 255, nothing)
cv2.createTrackbar('ThresholdLS', 'Test Parameters_Thresholding RGB', 0, 255, nothing)
cv2.createTrackbar('ThresholdUS', 'Test Parameters_Thresholding RGB', 0, 255, nothing)
cv2.createTrackbar('ThresholdLV', 'Test Parameters_Thresholding RGB', 0, 255, nothing)
cv2.createTrackbar('ThresholdUV', 'Test Parameters_Thresholding RGB', 0, 255, nothing)

cv2.namedWindow('Test Parameters_Effects')
cv2.createTrackbar('Dilate', 'Test Parameters_Effects', 1, 10, nothing)
cv2.createTrackbar('Erode', 'Test Parameters_Effects', 1, 10, nothing)
cv2.createTrackbar('MedianBlur', 'Test Parameters_Effects', 1, 10, nothing)

c = 'f'
while(cv2.waitKey!=-1 and c!=113):
    
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lh = cv2.getTrackbarPos('ThresholdLH','Test Parameters_Thresholding RGB')
    uh = cv2.getTrackbarPos('ThresholdUH','Test Parameters_Thresholding RGB')
    ls = cv2.getTrackbarPos('ThresholdLS','Test Parameters_Thresholding RGB')
    us = cv2.getTrackbarPos('ThresholdUS','Test Parameters_Thresholding RGB')
    lv = cv2.getTrackbarPos('ThresholdLV','Test Parameters_Thresholding RGB')
    uv = cv2.getTrackbarPos('ThresholdUV','Test Parameters_Thresholding RGB')
    
    dil = cv2.getTrackbarPos('Dilate','Test Parameters_Effects')
    ero = cv2.getTrackbarPos('Erode','Test Parameters_Effects')
    blr = cv2.getTrackbarPos('MedianBlur','Test Parameters_Effects')
    
    #Threshold for HSV range
    min_c = np.array([lh, ls, lv], np.uint8)
    max_c = np.array([uh, us, uv], np.uint8)
    mask = cv2.inRange(hsv_img, min_c, max_c)
    res = cv2.bitwise_and(im,im, mask= mask)
    cv2.imshow('Threshold_RGB',res)
    
    resized = cv2.dilate(res,np.ones((dil,dil),np.uint8))
    resized = cv2.erode(resized,np.ones((ero,ero),np.uint8))
    resized = cv2.medianBlur(resized,blr)
    cv2.imshow('Filtered_Image',resized)
    

    c = cv2.waitKey(1)

            
resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
            



for i in range(h):
    for j in range(w):
            
        if(resized[i,j]==0):
            ZZZ.append([])
            fill(resized,[i,j],255)
            count=count+1



for i in range(0,len(ZZZ)):
    ZZZ[i].sort(key = itemgetter(0,1))



for i in range(0 , len(ZZZ)):
    Q.append([])
    j=0
    k=0
    while j<len(ZZZ[i]):
        Q[i].append([])
        Q[i][k].append(ZZZ[i][j])
        l=j
        while(l+1<len(ZZZ[i])):
            if(not(ZZZ[i][l][0]==ZZZ[i][l+1][0] and ZZZ[i][l][1]+1==ZZZ[i][l+1][1])):
                break
            l=l+1
            if(l==len(ZZZ[i])):
                l=l-1
                break
        if(l==j):
            j=j+1
            k=k+1
        else:
            j=l
            Q[i][k].append(ZZZ[i][j])
            k=k+1
            j=j+1;
xprev = 0

for i in range(0,len(Q)-1):
        for j in range(0,len(Q[i])-1):
            if(len(Q[i][j])==2):
                xprev = Q[i][j][0][0]
    
if (state == dType.DobotConnect.DobotConnect_NoError):

    #Clean Command Queued
    dType.SetQueuedCmdClear(api)

    #Async Motion Params Setting
    dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1)
    dType.SetPTPCommonParams(api, 100, 100, isQueued = 1)
    count=0
    e=0
    water_dip()
    
    for i in range(13,len(Q)-1):
        for j in range(0,len(Q[i])-1):
            if(len(Q[i][j])==1):
                x1 = Q[i][j][0][0]
                y1 = Q[i][j][0][1]
                count=count+1;
                lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, TL_x-(float(x1)*img_size_h/float(h)), TL_y-(float(y1)*img_size_w/float(w)), zcord, 0, isQueued = 1)[0]
                im[y1,x1] = (0,255,0)
                cv2.imshow('Output',im) 

            else:
                x1 = Q[i][j][0][0]
                y1 = Q[i][j][0][1]
                x2 = Q[i][j][1][0]
                y2 = Q[i][j][1][1]


                xprev = x1
                e=e+1
                
                if (e%2==0):
                    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, TL_x-(float(x1)*img_size_h/float(h)), TL_y-(float(y1)*img_size_w/float(w)), zcord, 0, isQueued = 1)[0]
                    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, TL_x-(float(x2)*img_size_h/float(h)), TL_y-(float(y2)*img_size_w/float(w)), zcord, 0, isQueued = 1)[0]
                    count=count+2
                    cv2.line(im,(y1,x1),(y2,x2),(0,255,0),5)
                else:
                    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, TL_x-(float(x2)*img_size_h/float(h)), TL_y-(float(y2)*img_size_w/float(w)), zcord, 0, isQueued = 1)[0]
                    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, TL_x-(float(x1)*img_size_h/float(h)), TL_y-(float(y1)*img_size_w/float(w)), zcord, 0, isQueued = 1)[0]
                    count=count+2
                    cv2.line(im,(y2,x2),(y1,x1),(0,255,0),5)
                cv2.imshow('Output',im)
                
            if(count%25==0 or count%26==0):
                dType.SetQueuedCmdStartExec(api)

                #Wait for Executing Last Command 
                while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
                    dType.dSleep(100)
		
                #Stop to Execute Command Queued
                dType.SetQueuedCmdStopExec(api)
                dType.SetQueuedCmdClear(api)
                count=0
                water_dip()



    #Start to Execute Command Queued
    dType.SetQueuedCmdStartExec(api)
    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, 300, 80, zcord+10, 0, isQueued = 1)[0]
                    
    #Wait for Executing Last Command 
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(100)
		
    #Stop to Execute Command Queued
    dType.SetQueuedCmdStopExec(api)
    water_dip()
    
print("End")

#Disconnect Dobot
dType.DisconnectDobot(api)
cv2.destroyAllWindows()
cv2.waitKey(0)
