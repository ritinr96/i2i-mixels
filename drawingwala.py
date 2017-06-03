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



img = cv2.imread('Po3.jpg',-1)
img = cv2.resize(img,(400,300))

cap=cv2.imread('Po3.jpg',0)
resized=cv2.resize(cap,(400,300))
ret, resized = cv2.threshold(resized,60,255,cv2.THRESH_BINARY) 
cv2.imshow('asd',resized)
resized = cv2.dilate(resized,np.ones((3,3),np.uint8),50)
resized = cv2.erode(resized,np.ones((3,3),np.uint8),5)
#cv2.Canny(resized,0,100)
resized = cv2.medianBlur(resized,5)
cv2.imshow('asda',resized)

flag = 0
count=0
ZZZ=[]
YYY=0
Q = []

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
'''
for i in range(1):
    kernel = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    kernel= np.asanyarray(kernel)#Making the Laplacian + Original mask for image enhancement
    img_new = cv2.filter2D(resized,-1,kernel)#Applying the mask on every pixel to obtain the enhanced image

img_new = cv2.GaussianBlur(img_new,(5,5),0)
img_new = cv2.medianBlur(img_new,5)
ret,thresh1 = cv2.threshold(img_new,9,255,cv2.THRESH_BINARY_INV)
cv2.imshow('new',thresh1)
'''


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
    dType.SetPTPJumpParams(api,7,-40,isQueued=1)
    count=0
    e=0
    
    for i in range(0,len(Q)-1):
        for j in range(0,len(Q[i])-1):
            if(len(Q[i][j])==1):
                count=count+1;
                lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, 300-(float(Q[i][j][0][0])*50/float(300)), 80-(float(Q[i][j][0][1])*50/400), -55, 0, isQueued = 1)[0]
                img[Q[i][j][0][1],Q[i][j][0][0]] = (0,255,0)
                cv2.imshow('Output',img) 
            else:
                if (e%2==0):
                    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, 300-(float(Q[i][j][0][0])*50/float(300)), 80-(float(Q[i][j][0][1])*50/400), -55, 0, isQueued = 1)[0]
                    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 300-(float(Q[i][j][1][0])*50/float(300)), 80-(float(Q[i][j][1][1])*50/400), -55, 0, isQueued = 1)[0]
                    count=count+2
                    e=e+1
                    cv2.line(img,(Q[i][j][0][1],Q[i][j][0][0]),(Q[i][j][1][1],Q[i][j][1][0]),(0,255,0),5)
                else:
                    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPJUMPXYZMode, 300-(float(Q[i][j][1][0])*50/float(300)), 80-(float(Q[i][j][1][1])*50/400), -55, 0, isQueued = 1)[0]
                    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 300-(float(Q[i][j][0][0])*50/float(300)), 80-(float(Q[i][j][0][1])*50/400), -55, 0, isQueued = 1)[0]
                    count=count+2
                    e=e+1
                    cv2.line(img,(Q[i][j][1][1],Q[i][j][1][0]),(Q[i][j][0][1],Q[i][j][0][0]),(0,255,0),5)
                cv2.imshow('Output',img)      
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
