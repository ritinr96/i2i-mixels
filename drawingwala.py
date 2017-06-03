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


cap = cv2.VideoCapture(1)


flag = 0
count=0
ZZZ=[]
YYY=0
Q = []
zcord = -55.6

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


c = 'f'
cv2.namedWindow('test')
cv2.createTrackbar('Threshold', 'test', 0, 255, nothing)

while(cv2.waitKey!=-1 and c!=113):
    ret, frame = cap.read()
    im = cv2.resize(frame,(400,300))
    frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(frame2,(400,300))

    resized=cv2.resize(img,(400,300))
    thr = cv2.getTrackbarPos('Threshold','test')
    ret, resized = cv2.threshold(resized,thr,255,cv2.THRESH_BINARY) 
    cv2.imshow('asd',resized)
    resized = cv2.dilate(resized,np.ones((3,3),np.uint8),5)
    resized = cv2.erode(resized,np.ones((3,3),np.uint8),5)
    #cv2.Canny(resized,0,100)
    resized = cv2.medianBlur(resized,3)
    cv2.imshow('test',resized)
    c = cv2.waitKey(1)
    print c

cv2.imwrite('Pic_Pose.jpg',resized)


for i in range(300):
    for j in range(400):
            
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

            
                   

if (state == dType.DobotConnect.DobotConnect_NoError):

    #Clean Command Queued
    dType.SetQueuedCmdClear(api)

    #Async Motion Params Setting
    dType.SetHOMEParams(api, 200, 0, 136, 0, isQueued = 1)
    dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1)
    dType.SetPTPCommonParams(api, 100, 100, isQueued = 1)
    dType.SetPTPJumpParams(api,4,-40,isQueued=1)
    count=0
    e=0
    
    for i in range(0,len(Q)-1):
        for j in range(0,len(Q[i])-1):
            if(len(Q[i][j])==1):
                count=count+1;
                lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, 300-(float(Q[i][j][0][0])*50/float(300)), 80-(float(Q[i][j][0][1])*50/400), zcord, 0, isQueued = 1)[0]
                im[Q[i][j][0][1],Q[i][j][0][0]] = (0,255,0)
                cv2.imshow('Output',im) 
            else:
                if (e%2==0):
                    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, 300-(float(Q[i][j][0][0])*50/float(300)), 80-(float(Q[i][j][0][1])*50/400), zcord, 0, isQueued = 1)[0]
                    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 300-(float(Q[i][j][1][0])*50/float(300)), 80-(float(Q[i][j][1][1])*50/400), zcord, 0, isQueued = 1)[0]
                    count=count+2
                    e=e+1
                    cv2.line(im,(Q[i][j][0][1],Q[i][j][0][0]),(Q[i][j][1][1],Q[i][j][1][0]),(0,255,0),5)
                else:
                    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, 300-(float(Q[i][j][1][0])*50/float(300)), 80-(float(Q[i][j][1][1])*50/400), zcord, 0, isQueued = 1)[0]
                    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 300-(float(Q[i][j][0][0])*50/float(300)), 80-(float(Q[i][j][0][1])*50/400), zcord, 0, isQueued = 1)[0]
                    count=count+2
                    e=e+1
                    cv2.line(im,(Q[i][j][1][1],Q[i][j][1][0]),(Q[i][j][0][1],Q[i][j][0][0]),(0,255,0),5)
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



    #Start to Execute Command Queued
    dType.SetQueuedCmdStartExec(api)
    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, 300, 80, zcord+10, 0, isQueued = 1)[0]
                    
    #Wait for Executing Last Command 
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(100)
		
    #Stop to Execute Command Queued
    dType.SetQueuedCmdStopExec(api)
    
print("End")
#Disconnect Dobot
dType.DisconnectDobot(api)


cv2.imshow("asdad",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
