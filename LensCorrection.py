# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 10:03:31 2018

@author: raman
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir
import random

def getCircleCenters(image, drawLines = True):

    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    retval, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
#    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #image = cv2.dilate(image, el, iterations=6)
    
    #cv2.imwrite("dilated.png", image)
    
    temp, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    cur_imgpoints = []
    cur_objpoints = []
    radii = []
    distances = []
    
    pitch = 55/0.616736797576255
    vertPitch = (pitch**2 - pitch**2/4)**0.5
    
    for contour in contours:
        area = cv2.contourArea(contour)
    
        # there is one contour that contains all others, filter it out
        if area > 500:
            continue
    
        br = cv2.boundingRect(contour)
        radii.append(br[2])
    
        m = cv2.moments(contour)

        center = (np.float32(m['m10'] / m['m00']), np.float32(m['m01'] / m['m00']))
        cur_imgpoints.append(center)
        
        x, y = center[0], center[1]        
        distances.append(((center[0]-image.shape[1]/2)**2 + (center[1]-image.shape[0]/2)**2)**0.5)
        
   
    print("There are {} circles".format(len(cur_imgpoints)))
    
    cur_imgpoints = np.array(cur_imgpoints)
    
    def rotatePoint(point, origin, angle):
        ox, oy = origin
        px, py = point
    
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy
    
    def hasCounterPart(point, otherpoints, threshold):
        distances = otherpoints - np.array([point[0], point[1]])
        distances = [(abs(i[0])+abs(i[1])) for i in distances]
        if min(distances) < threshold:
            return True, distances.index(min(distances))
        else:
            return False, distances.index(min(distances))
    
    #select most important points...
    indexClosestToCenter = np.argmin(distances)
    offsetRelevantPoints = []

    for i in range(5):
        offsetRelevantPoints.append(cur_imgpoints[indexClosestToCenter-2+i])

    #get rotation
    rotAngle = np.arctan((offsetRelevantPoints[-1][1] - offsetRelevantPoints[0][1]) /
                         (offsetRelevantPoints[-1][0] - offsetRelevantPoints[0][0]))
    
    centerX, centerY = cur_imgpoints[indexClosestToCenter][0], cur_imgpoints[indexClosestToCenter][1]
    
    #guess row and colindex:
    centerCol = np.round(centerX/pitch)
    centerRow = np.round(centerY/vertPitch)
    numCols = int(2*centerCol+2)
    numRows = int(2*centerCol+1)
    
    startX = centerX - (centerCol+1)*pitch
    startY = centerY - (centerRow+1)*vertPitch
    
    for row in range(numRows):
        for col in range(numCols):

            if row%2 == 0:
                curX = col*pitch + startX
            else:
                curX = col*pitch + startX - pitch/2  
            curY = row*vertPitch + startY
            curX, curY = rotatePoint((curX, curY), (centerX, centerY), rotAngle)
 
            if curX >= 0 and curY >= 0 and curX < image.shape[1] and curY < image.shape[0]:   #in image boundary?
                #has an imgpoint counterpart?
                if hasCounterPart((curX, curY), cur_imgpoints, 20)[0]:
                    cur_objpoints.append((np.float32(curX), np.float32(curY), np.float32(0)))
    
    cur_objpoints = np.array(cur_objpoints)
    
    #output image
    if len(cur_imgpoints) > 0:
#        radius = int(np.average(radii)) + 5
        
        for index, center in enumerate(cur_imgpoints):
            center = (int(center[0]), int(center[1]))
            cv2.circle(image, center, 3, (255, 0, 0), -1)
#            cv2.circle(image, center, radius, (0, 255, 0), 1)
            
            if index > 0 and drawLines:
                index = int(index)
                cv2.line(img, center, tuple(cur_imgpoints[index-1]), (128, 128, 0))

    #reorder obj_points
    cur_imgpoints = [list(i) for i in cur_imgpoints]           #convert to list -> makes deleting of elements by remove method easy
    final_obj_points = []
    remove = []
    for index, imgpoint in enumerate(cur_imgpoints):
        valid, closestIndex = hasCounterPart(imgpoint, cur_objpoints[:, :2], 20)
        if valid:
            final_obj_points.append(cur_objpoints[closestIndex, :])
        else:
            remove.append(imgpoint)
    for i in remove:
        cur_imgpoints.remove(i)
    
    #select n random points
#    randomIndices = random.sample(list(np.arange(0, len(cur_imgpoints))), 10)
#    cur_imgpoints = [point for index, point in enumerate(cur_imgpoints) if index in randomIndices]
#    final_obj_points = [point for index, point in enumerate(final_obj_points) if index in randomIndices]
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            k = i+j**2 + 0.531**4 + 0.38947**6
            k = i+j**2 + 0.531**4 + 0.38947**6
    final_obj_points = np.array(final_obj_points)
    
#    #artificially stretch points...
#    final_obj_points = final_obj_points - np.array([376, 240, 0])
#    final_obj_points = final_obj_points * 1.1
#    final_obj_points = final_obj_points + np.array([376, 240, 0])
    
#    for i in range(len(final_obj_points)):
#        final_obj_points[i, 0] += np.random.rand()*30 
#        final_obj_points[i, 1] += np.random.rand()*30
    
    
    for point in final_obj_points:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), 10, (0, 255, 128), 2)

    
    
    cur_imgpoints = np.array(cur_imgpoints)
    
    assert len(cur_imgpoints) == len(final_obj_points), 'len(cur_imgpoints) ({}) != len(final_obj_points) ({})'.format(len(cur_imgpoints), len(final_obj_points))
    
    return image, cur_imgpoints, np.float32(final_obj_points)

imgs = []
objpoints = []
imgpoints = []

#get images...
files = listdir()
for index, fname in enumerate(files):
    if fname.find('.bmp') > -1:
#        fname = '{} Filter.bmp'.format('0'+str(i+1) if (i+1) < 10 else str(i+1))
        print('processesing', fname)
        img = cv2.imread(fname)
        foundCircles, cur_imgpoints, cur_objpoints = getCircleCenters(img, drawLines=False)
        
        if index == 0:
#            plt.subplot(221)
            plt.imshow(foundCircles)
        
        imgpoints.append(cur_imgpoints)
        objpoints.append(cur_objpoints)
        imgs.append(foundCircles)

#Run the Camera Calibration, obtain camera matrix, distortion coefficients, rotation and translation vectors 
imgpoints = np.array(imgpoints)
objpoints = np.array(objpoints)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgs[0].shape[:2], None, None)

#
####get new CameraMatrix
#gridimg = np.zeros(img.shape[:2])
#line_dist = 20
#for col in range(gridimg.shape[1]//line_dist):
#    cv2.line(gridimg, (col*line_dist+5, 0+5), (col*line_dist+5, gridimg.shape[0]+5), 1, thickness = 2)
#
#for row in range(gridimg.shape[0]//line_dist):
#    cv2.line(gridimg, (0+5, row*line_dist+5), (gridimg.shape[1]+5, row*line_dist+5), 1, thickness = 2)
#
#img = gridimg
#
#h,  w = img.shape[:2]
#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
#
###UNDISTORT!!! =)
#dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
##mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
##dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
#
#
#
#
#np.savetxt('matrix.txt', mtx)
#np.savetxt('distances.txt', dist)
#np.savetxt('cameramatrix.txt', newcameramtx)
#np.savetxt('roi.txt', roi)
#
## crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
#
#
#plt.subplot(223)
#plt.imshow(gridimg, cmap = 'gray')
#plt.subplot(224)
#plt.imshow(dst, cmap = 'gray')
#plt.tight_layout()