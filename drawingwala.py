import cv2
import time
import threading
import DobotDllType as dType
import numpy as np
from operator import itemgetter

cap=cv2.imread('yo10.jpg',0)
cv2.imshow('asdsa',cap)
ret,thresh1 = cv2.threshold(cap,60,255,cv2.THRESH_BINARY)

resized=cv2.resize(thresh1,(160,100))
cv2.imshow('asd',resized)

count=0
ZZZ=[]
YYY=0
def fill(resized, start_coords, fill_value):
    global YYY 
    """
    Flood fill algorithm
    
    Parameters
    ----------
    resized : (M, N) ndarray of uint8 type
        Image with flood to be filled. Modified inplace.
    start_coords : tuple
        Length-2 tuple of ints defining (row, col) start coordinates.
    fill_value : int
        Value the flooded area will take after the fill.
        
    Returns
    -------
    None, ``resized`` is modified inplace.
    """
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



for i in range(100):
    for j in range(160):
            
        if(resized[i,j]==0):
            ZZZ.append([])
            fill(resized,[i,j],255)
            count=count+1


for i in range(0,len(ZZZ)):
    ZZZ[i].sort(key = itemgetter(0,1))
    print ZZZ[i]
        

cv2.imshow("asdad",resized)
cv2.waitKey(0)
