import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

bMouseClick = False
LastClickPosition = (0,0)
WindowScale = 0.0


#################################################################################################################################################################################################
### Mouse Click Events and Query ################################################################################################################################################################
# ╰┈➤ Track mouse clicks:

def click_event(event, x, y, flags, param):
    global bMouseClick
    global LastClickPosition
    global WindowScale

    if event == cv2.EVENT_LBUTTONDOWN:
        #print(f'Mouse click at ({x}, {y})')
        bMouseClick = True
        LastClickPosition = (x * WindowScale, y * WindowScale)       

def resetClickQuery():
    global bMouseClick
    bMouseClick = False

def QueryMouseClick():
    global bMouseClick
    global LastClickPosition
    
    if (bMouseClick):
        bMouseClick = False
        return LastClickPosition
    else: return (None, None)


#################################################################################################################################################################################################
### Display Image ###############################################################################################################################################################################
# ╰┈➤ Display an image to the provided window, scaled to the provided resolution: 
  
def displayImage(window, img, display_res):
    ### Resize Image ---------------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Resize image to desired display resolution, without changing aspect ratio:
    if (img.shape[0] > display_res[0] or img.shape[1] > display_res[1]):
        h_scale = img.shape[0] / display_res[0]
        w_scale = img.shape[1] / display_res[1]
    
        scale = max(h_scale, w_scale)
        new_dim = [int(img.shape[1] / scale), int(img.shape[0] / scale)]
        img = cv2.resize(img, new_dim)
        
        global WindowScale
        WindowScale = scale
        
    cv2.imshow(window, img)
    cv2.setMouseCallback(window, click_event)  #Set the callback function for mouse events

#################################################################################################################################################################################################
### Gamma Correction ############################################################################################################################################################################
# ╰┈➤ Apply the desired gamma correction to the provided image. >1 increases the brightness, <1 decreases the brightness.
     
def applyGammaCorrection(img, gamma):
    img_gamma_correct = np.array(255*(img/255) ** (1/gamma), dtype = 'uint8')
    return img_gamma_correct



#################################################################################################################################################################################################
### Extract Image Cutout Using Rect #############################################################################################################################################################
# ╰┈➤ Extract a cutout of the provided image using the provided rectangle. Optionally, the extracted image can be resized to the provided width.

def extractImgUsingRect(img, rect, target_width = -1):
    
    ### Convert to box -------------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Convert to box to obtain corner points on source image:
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    src_pts = box.astype("float32")
    
    ### Find width and height ------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Find width and height to derive corner points on destination image
    w = int(rect[1][0])
    h = int(rect[1][1])
    
    ### Enforce Horizontal Alignment -----------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Ensure the width is greater than the height, if not, rotate the rectangle by 90 degrees:
    
    if w < h:                                                                           #If width is less than height:
        w, h = h, w                                                                         #Swap width and height
        src_pts = np.roll(src_pts, 1, axis=0)                                               #Rotate source rectangle by 90 degrees
    
    ### Extract Cutout ------------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Define destination corner points and extract cutout using perspective transformation:
        
    dst_pts = np.array([[0, h-1],                                                       #Define destination corner points using width and height
                        [0, 0],
                        [w-1, 0],
                        [w-1, h-1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)                                   #Obtain perspective transformation matrix
    extracted = cv2.warpPerspective(img, M, (w, h))                                     #Apply perspective transformation to extract cutout
    
    ### Apply Target Width --------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ If provided a target width, resize the extracted cutout to the desired width:
    if (target_width > 0):
        scale = w / target_width
        target_height = h / scale
        target_dim = [int(target_width), int(target_height)]
        extracted = cv2.resize(extracted, target_dim)
    
    return extracted



#################################################################################################################################################################################################
### Extract Dominat Colour ######################################################################################################################################################################
# ╰┈➤ Extract the dominant colour from the provided image using k-means clustering. Optionally, the minimum value of the V channel can be enforced.

def ExtractDominantColour(img, k_clusters, v_min=-1):
    Z = img.reshape((-1,3))                                                             #Reshape to 2D array
    Z = np.float32(Z)                                                                   #Convert to float32
 
                                                                                    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)            #Define termination criteria
    _,labels,centers = cv2.kmeans(Z,k_clusters,None,
                                  criteria,10,cv2.KMEANS_RANDOM_CENTERS)                #Apply k-means clustering
    
    if (k_clusters == 1):                                                               #If only one cluster:
        dominant_colour = centers[0]                                                        #Return the only centre (colour)
        return dominant_colour
    
    most_common_centroid_index = np.argmax(np.bincount(labels.flatten()))               #Find the most frequent label
    dominant_colour = centers[most_common_centroid_index]                               #Obtain the centre (colour) corresponding to the most frequent label
    
    if (v_min > 0):                                                                     #If a minimum V value is provided:
        dominant_colour_hsv = cv2.cvtColor(np.uint8([[dominant_colour]]), cv2.COLOR_BGR2HSV)[0][0]
        if dominant_colour_hsv[2] < v_min:                                                  #If the HSV 'V' value is below the minimum:
            print("Dominant colour too dark, finding second most common colour")
            second_most_common_centroid_index = np.argsort(np.bincount(labels.flatten()))[-2]   #Find the second most frequent label
            dominant_colour = centers[second_most_common_centroid_index]                        #Obtain the centre (colour) corresponding to the second most frequent label
                  
    #centers = np.uint8(centers)                                                        #Obtain and display quantised image if desired for debug
    #res = centers[labels.flatten()]
    #preview = res.reshape((img.shape))
    #cv2.imshow("Preview", preview)
    #cv2.waitKey(0) # waits for user to press any key 
    
    #print(cv2.cvtColor(np.uint8([[dominant_colour]]), cv2.COLOR_BGR2HSV)[0][0])
    return dominant_colour



#################################################################################################################################################################################################
### Locate Missing Band #########################################################################################################################################################################
# ╰┈➤ Locate the missing band in the provided image using the provided range and target band width. Optionally, the HSV channel and threshold factor can be adjusted.

def LocateMissingBand(xm, rng, target_band_width, abs_diff_img, HSV_Channel=2, Threshold_Factor=0.8):
    
    h = abs_diff_img.shape[0]                                                           #Obtain the image height
    
    x0 = xm - int(rng/2)                                                                #Overestimate band start and end
    x1 = xm + int(rng/2)
                
    diff_HSV = cv2.cvtColor(abs_diff_img[:, x0:x1], cv2.COLOR_BGR2HSV)                  #Recalulate band mask from abs. diff. image about predicted band location (rather than global)
    otsu_threshold, _ = cv2.threshold(diff_HSV[:, :, HSV_Channel], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)                 #Given a smaller range, the automatic OTSU threshold...
    _, mask = cv2.threshold(diff_HSV[:, :, HSV_Channel], int(otsu_threshold * Threshold_Factor), 255, cv2.THRESH_BINARY)        #...should more accurately segment the subtle band
    #cv2.imshow("Preview", mask)
                
    col_sum = np.sum(mask, axis = 0)                                                    #Sum the mask columns
    x_max = np.argmax(col_sum)                                                          #Find the mask column with largest sum
                   
    continuity = col_sum[x_max]/(h*255)                                                 #Find the continuity of the max column (0-1)
    
    if (continuity == 1.0):                                                             #If the continuity is perfect, find the range it remains perfect
        x_max_end = x_max
        for i in range(x_max,len(col_sum)):                                                 #Iterate starting from the max column to the end
            cont = col_sum[i]/(h*255)
            if (cont < 1.0):                                                                    #Search until the continuity drops below 1.0:
                x_max_end = i                                                                       #Record the end of the perfect continuity range
                break                                                                               #Exit the loop
        
        if (x_max_end - x_max > target_band_width):                                         #If the perfect continuity range is wider than the original target, return this as the new band                  
            x_max = x0 + x_max                                                                  #Translate from search-range index to image index 
            x_max_end = x0 + x_max_end
            
            band = [x_max, x_max_end, x_max_end - x_max]                                        #Create and return the new band info
            return (band, continuity)
        else: x_max = int((x_max + x_max_end)/2)                                            #Otherwise, find the center of the perfect continuity range
                
    xm = x0 + x_max                                                                     #Translate max intensity index from search-range index to image index
                    
    x0 = xm - int(target_band_width/2)                                                  #Calculate the new band start and end
    x1 = xm + int(target_band_width/2)
                    
    band = [x0, x1, target_band_width]                                                  #Create and return the new band info
    return (band, continuity)



#################################################################################################################################################################################################
### Convert Int to Metric String ################################################################################################################################################################     
# ╰┈➤ Convert an integer to a metric string, e.g. 1000 -> '1k', 1000000 -> '1M', 1500 -> '1k5
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
    elif n < 10:
        # If the number is less than 1000, format it as an integer
        s = str(n)
    else:
        # If the number is less than 1000, format it as an integer
        s = str(int(n))
    return s



#################################################################################################################################################################################################
### E-Series Verification #######################################################################################################################################################################     
# ╰┈➤ Check if provided resistance number belongs to the E24, or E96 series

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
### Display Bounding Rect #######################################################################################################################################################################     
# ╰┈➤ Outline a bounding box on the provided image:
def Display_Bounding(display_image, bounding_rect, colour, thickness):
    box = cv2.boxPoints(bounding_rect)                                                          #Find box points from rectangle
    box = np.intp(box)                     
    
    cv2.drawContours(display_image,[box],0,colour,thickness)                                    #Draw bounding box on mask

    return box

def PIL_writeText(img, text, position, colour, fontsize):
    # Convert the image to RGB (OpenCV uses BGR)
    cv2_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform the cv2 image to PIL
    pil_img = Image.fromarray(cv2_img_rgb)

    draw = ImageDraw.Draw(pil_img)

    # Use a truetype font
    font = ImageFont.truetype("data/cambriab.ttf", fontsize, encoding="unic")

    # Draw text
    draw.text(position, text, font=font, fill=colour)

    # Get back the image to OpenCV
    cv2_img_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    return cv2_img_text

        
def Append_To_Object_Log(img, log, offset=0, padding=0, bool_convBGR=False, bool_upwards=False):
    
    if (bool_convBGR): img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w, _ = img.shape

    h_log, w_log, _ = log.shape
        
    if (offset > 0) or (padding > 0): img = cv2.copyMakeBorder(img, 0, padding, offset, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    if (w + offset < w_log): img = cv2.copyMakeBorder(img, 0, 0, 0, w_log - (w + offset), cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
    if (bool_upwards):
        log = np.concatenate((img, log), axis=0)
    else:
        log = np.concatenate((log, img), axis=0)
            
    return log

def Construct_Classification_Preview(extracted_band_classes, target_resistor_width, colour_lookup, colour_BGR_lookup):
    
    num_bands = len(extracted_band_classes)
    preview_width = int((target_resistor_width - 30)/num_bands)
    
    blank_img = np.zeros((50, preview_width, 3), np.uint8)        
    preview_nums = np.zeros((50, 15, 3), np.uint8)
    for j in range(num_bands):
        text_img = blank_img.copy()
        textsize = cv2.getTextSize(str(extracted_band_classes[j]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2)[0]
        text_posX = int(preview_width/2) - int(textsize[0]/2)
        text_posY = 25 + int(textsize[1]/2)
        cv2.putText(text_img, str(extracted_band_classes[j]), (text_posX, text_posY), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(200, 200, 255), thickness=2, lineType=cv2.LINE_AA)
            
        text_img[4:95, 3:6] = (255, 255, 255)
        text_img[6:93, 6:11] = colour_BGR_lookup[extracted_band_classes[j]]
        text_img[6:93, preview_width-12:preview_width-7] = colour_BGR_lookup[extracted_band_classes[j]]
        text_img[4:95, preview_width-7:preview_width-4] = (255, 255, 255)

        preview_nums = np.concatenate((preview_nums, text_img), axis=1)
    preview_nums = np.concatenate((preview_nums, np.zeros((50, 15, 3), np.uint8)), axis=1)
        
    blank_img = np.zeros((preview_width, 100, 3), np.uint8)        
    preview_colours = np.zeros((100, 15, 3), np.uint8)
    for j in range(num_bands):
        text_img = blank_img.copy()
        textsize = cv2.getTextSize(colour_lookup[extracted_band_classes[j]], cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2)[0]
        text_posX = 50 - int(textsize[0]/2)
        text_posY = int(preview_width/2) + int(textsize[1]/2) - 1
        cv2.putText(text_img, colour_lookup[extracted_band_classes[j]], (text_posX, text_posY), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(200, 200, 255), thickness=2, lineType=cv2.LINE_AA)

        #Roll image 90deg CCW
        text_img = cv2.rotate(text_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        text_img[4:95, 3:6] = (255, 255, 255)
        text_img[6:93, 6:11] = colour_BGR_lookup[extracted_band_classes[j]] #(80, 80, 80)
        text_img[6:93, preview_width-12:preview_width-7] = colour_BGR_lookup[extracted_band_classes[j]]
        text_img[4:95, preview_width-7:preview_width-4] = (255, 255, 255)

        preview_colours = np.concatenate((preview_colours, text_img), axis=1)
    preview_colours = np.concatenate((preview_colours, np.zeros((100, 15, 3), np.uint8)), axis=1)

    return (preview_nums, preview_colours)
    
def Construct_Inverted_Resistance_Preview(inverted_resistance_string, bAmbiguousReadDirection, target_resistor_width):
    
    preview_inverted_resistance = np.zeros((125, target_resistor_width, 3), np.uint8)        
            
    textsize = cv2.getTextSize("Inverted Reading:", cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, thickness=2)[0]
    text_posX = 15
    text_posY = 15 + textsize[1]
    cv2.putText(preview_inverted_resistance, "Inverted Reading:", (text_posX, text_posY), cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(210, 210, 255), thickness=1, lineType=cv2.LINE_AA)

    textsize = cv2.getTextSize(inverted_resistance_string, cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, thickness=2)[0]
    text_posX = target_resistor_width - textsize[0] - 45
    text_posY = 30 + 2*textsize[1]
    cv2.putText(preview_inverted_resistance, inverted_resistance_string, (text_posX, text_posY), cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 0, 200), thickness=2, lineType=cv2.LINE_AA)

    textsize = cv2.getTextSize("Ambiguous?", cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, thickness=2)[0]
    text_posX = 15
    text_posY = 55 + 3*textsize[1]
    cv2.putText(preview_inverted_resistance, "Ambiguous?", (text_posX, text_posY), cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(210, 210, 255), thickness=1, lineType=cv2.LINE_AA)

    display_string = "Yes" if bAmbiguousReadDirection else "No"
    textsize = cv2.getTextSize(display_string, cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, thickness=2)[0]
    text_posX = target_resistor_width - textsize[0] - 40
    text_posY = 55 + 3*textsize[1]
    cv2.putText(preview_inverted_resistance, display_string, (text_posX, text_posY), cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0, 0, 200), thickness=2, lineType=cv2.LINE_AA)
    
    preview_inverted_resistance[-4:-1, :] = [255, 255, 255]

    return preview_inverted_resistance

