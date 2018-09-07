# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:09:08 2018

@author: brandt
"""
import cv2
import numpy as np

def getImageAndObjectPoints(image):

    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    retval, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    temp, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    cur_imgpoints = []
    cur_objpoints = []
    distances = []
    
    pitch = 55/0.616736797576255
    vertPitch = (pitch**2 - pitch**2/4)**0.5
    
    for contour in contours:
        area = cv2.contourArea(contour)
    
        # there is one contour that contains all others, filter it out
        if area > 500:
            continue
    
        m = cv2.moments(contour)
        center = (np.float32(m['m10'] / m['m00']), np.float32(m['m01'] / m['m00']))
        cur_imgpoints.append(center)
#        x, y = center[0], center[1]        
        distances.append(((center[0]-image.shape[1]/2)**2 + (center[1]-image.shape[0]/2)**2)**0.5)

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
        for index, center in enumerate(cur_imgpoints):
            center = (int(center[0]), int(center[1]))
            cv2.circle(image, center, 3, (255, 0, 0), -1)


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
    
    final_obj_points = np.array(final_obj_points)
   
    for point in final_obj_points:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), 10, (0, 255, 128), 2)

    cur_imgpoints = np.array(cur_imgpoints)
    
    assert len(cur_imgpoints) == len(final_obj_points), 'len(cur_imgpoints) ({}) != len(final_obj_points) ({})'.format(len(cur_imgpoints), len(final_obj_points))
    
    return image, cur_imgpoints, np.float32(final_obj_points)