from calendar import c
from pyexpat.model import XML_CTYPE_MIXED
import cv2
import numpy as np # matrix manipulations

from matplotlib import pyplot as plt # this lets you draw inline pictures in the notebooks
import pylab # this allows you to control figure size 

def displayImage(window, img, display_res):
    ### Resize Image ---------------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Resize image to desired display resolution, without changing aspect ratio:
    if (img.shape[0] > display_res[0] or img.shape[1] > display_res[1]):
        h_scale = img.shape[0] / display_res[0]
        w_scale = img.shape[1] / display_res[1]
    
        scale = max(h_scale, w_scale)
        new_dim = [int(img.shape[1] / scale), int(img.shape[0] / scale)]
        img = cv2.resize(img, new_dim) 
        
    cv2.imshow(window, img)

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

def belongs_to_E24(resistance_num):
    E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]
    
    while (resistance_num >= 10): resistance_num /= 10.0
    while (resistance_num < 1.0): resistance_num *= 10.0
    
    resistance_num = round(resistance_num, 2)
    
    return resistance_num in E24
    
def belongs_to_E96(resistance_num):
    E96 = [1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30, 1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09, 3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12, 4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49, 5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32, 7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76, 10.00]
    
    while (resistance_num >= 10): resistance_num /= 10.0
    while (resistance_num < 1.0): resistance_num *= 10.0
    
    resistance_num = round(resistance_num, 2)
    
    return resistance_num in E96


#################################################################################################################################################################################################
### Display Boudning Rect #######################################################################################################################################################################     
# ╰┈➤ Outline a bounding box on the provided image:
def Display_Bounding(display_image, bounding_rect, colour, thickness):
    box = cv2.boxPoints(bounding_rect)                                                  #Find box points from rectangle
    box = np.intp(box)                     
    
    cv2.drawContours(display_image,[box],0,colour,thickness)                                   #Draw bounding box on mask

    return box