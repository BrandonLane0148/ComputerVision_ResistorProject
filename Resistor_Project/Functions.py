from calendar import c
from pyexpat.model import XML_CTYPE_MIXED
import cv2
import numpy as np # matrix manipulations

from matplotlib import pyplot as plt # this lets you draw inline pictures in the notebooks
import pylab # this allows you to control figure size 

def applyGammaCorrection(img, gamma):
    img_gamma_correct = np.array(255*(img/255) ** (1/gamma), dtype = 'uint8')
    return img_gamma_correct

def extractImgUsingRect(img, rect, target_width = -1):
    
    #Convert to box to obtain corner points on source image
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    src_pts = box.astype("float32")
    
    #Find width and height to derive corner points on destination image
    w = int(rect[1][0])
    h = int(rect[1][1])
    
    # Check if the width is less than the height
    if w < h:
        # Swap width and height
        w, h = h, w
        # Rotate source rectangle by 90 degrees
        src_pts = np.roll(src_pts, 1, axis=0)
        
    dst_pts = np.array([[0, h-1],
                        [0, 0],
                        [w-1, 0],
                        [w-1, h-1]], dtype="float32")
    
    #Obtain perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    #Directly warp the rotated rectangle to get the straightened rectangle
    extracted = cv2.warpPerspective(img, M, (w, h))
    
    if (target_width > 0):
        scale = w / target_width
        target_height = h / scale
        target_dim = [int(target_width), int(target_height)]
        extracted = cv2.resize(extracted, target_dim)
    
    return extracted

def ExtractDominantColour(img, k_clusters, v_min=-1):
    Z = img.reshape((-1,3))
    Z = np.float32(Z) # convert to np.float32
 
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _,labels,centers = cv2.kmeans(Z,k_clusters,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    if (k_clusters == 1):
        dominant_colour = centers[0]
        return dominant_colour
    
    most_common_centroid_index = np.argmax(np.bincount(labels.flatten()))
    dominant_colour = centers[most_common_centroid_index]
    
    if (v_min > 0):
        dominant_colour_hsv = cv2.cvtColor(np.uint8([[dominant_colour]]), cv2.COLOR_BGR2HSV)[0][0]
        if dominant_colour_hsv[2] < v_min:
            # Find the second most frequent label
            print("Dominant colour too dark, finding second most common colour")
            second_most_common_centroid_index = np.argsort(np.bincount(labels.flatten()))[-2]
            dominant_colour = centers[second_most_common_centroid_index]
    
    #Obtain and display quantised image if desired for debug
    #centers = np.uint8(centers)
    #res = centers[labels.flatten()]
    #preview = res.reshape((img.shape))
    #cv2.imshow("Preview", preview)
    #cv2.waitKey(0) # waits for user to press any key 
    
    #print(cv2.cvtColor(np.uint8([[dominant_colour]]), cv2.COLOR_BGR2HSV)[0][0])
    return dominant_colour

def LocateMissingBand(xm, rng, target_band_width, abs_diff_img, HSV_Channel=2, Threshold_Factor=0.8):
    #Obtain the image height
    h = abs_diff_img.shape[0] 
    
    #Overestimate band start and end
    x0 = xm - int(rng/2)
    x1 = xm + int(rng/2)
                
    #Recalulate band mask about predicted band location
    diff_HSV = cv2.cvtColor(abs_diff_img[:, x0:x1], cv2.COLOR_BGR2HSV)
    otsu_threshold, _ = cv2.threshold(diff_HSV[:, :, HSV_Channel], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(diff_HSV[:, :, HSV_Channel], int(otsu_threshold * Threshold_Factor), 255, cv2.THRESH_BINARY)
    #cv2.imshow("Preview", mask)
                
    #Find mask column with largest average value
    col_sum = np.sum(mask, axis = 0)
    x_max = np.argmax(col_sum)
                   
    #Find the continuity of the max column
    continuity = col_sum[x_max]/(h*255)
    
    #If the continuity is perfect, find the range it remains perfect
    if (continuity == 1.0):
        x_max_end = x_max
        #Search until the continuity drops below 1.0
        for i in range(x_max,len(col_sum)):
            cont = col_sum[i]/(h*255)
            if (cont < 1.0):
                x_max_end = i
                break
        #If the perfect continuity range is wider than the original target, return this as the new band
        if (x_max_end - x_max > target_band_width):
            #Translate from range index to image index
            x_max = x0 + x_max
            x_max_end = x0 + x_max_end
            
            #Create and return the new band info
            band = [x_max, x_max_end, x_max_end - x_max]
            return (band, continuity)
        
        #Otherwise, find the center of the perfect continuity range
        else: x_max = int((x_max + x_max_end)/2)      
                
    #Translate from range index to image index
    xm = x0 + x_max
                    
    #Calculate the new band start and end
    x0 = xm - int(target_band_width/2)
    x1 = xm + int(target_band_width/2)
                    
    #Create and return the new band info
    band = [x0, x1, target_band_width]
    return (band, continuity)

def int_to_metric_string(n):
    # Define the metric prefixes
    metric_prefixes = ['', 'k', 'M', 'G', 'T', 'P']

    # Find the appropriate metric prefix
    for i in range(len(metric_prefixes)):
        if abs(n) < 1000:
            break
        n /= 1000

    # Format the number as a string
    if i > 0 and n == int(n):
        # If the number is an integer, format it without a decimal point
        s = str(int(n)) + metric_prefixes[i]
    elif i > 0:
        # If the number is a decimal, format it with the decimal point replaced by the next metric prefix
        s = str(n).replace('.', metric_prefixes[i])
    else:
        # If the number is less than 1000, format it as an integer
        s = str(int(n))

    return s
