import cv2 #opencv itself
import numpy as np # matrix manipulations
from tensorflow.keras.models import load_model

import Functions as func

#################################################################################################################################################################################################
### Resistor Segmentation #######################################################################################################################################################################
# ╰┈➤ Isolate the resistors from the background:

def Resistor_Segmentation(filtered_input_image):

    ### Gamma Correct --------------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Increase gamma to drown out background saturation:
    image_gamma_correct =  func.applyGammaCorrection(filtered_input_image, 3) #consider opposite for dark background?

    ### OTSU Threshold -------------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Automatic threshold on HSV 'S' Channel to isolate coloured (saturated) resistor body from background:
    image_HSV = cv2.cvtColor(image_gamma_correct, cv2.COLOR_BGR2HSV)
    otsu_threshold, mask_OTSU = cv2.threshold(image_HSV[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)

    ### Mask Erosion ---------------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Erode Mask to remove wires and noise:
    erosion_dist = 1
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_dist + 1, 2 * erosion_dist + 1))
    mask_eroded = cv2.erode(mask_OTSU, erosion_kernel, iterations = 2)

    ### Mask Dilation --------------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Dilate Mask to shrink gaps:
    dilation_dist = 1
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_dist + 1, 2 * dilation_dist + 1))
    mask_dilated = cv2.dilate(mask_eroded, dilation_kernel, iterations = 2)


    return mask_dilated



#################################################################################################################################################################################################
### Resistor Extraction #########################################################################################################################################################################
# ╰┈➤ Extract individual resistors to seperate images from the segmentation mask:

def Resistor_Extraction(segmentation_mask, filtered_input_image, min_resistor_res, target_resistor_width):
    
    ### Gaussian Blur --------------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Blur mask to merge nearby outlines:
    mask_blur = cv2.GaussianBlur(segmentation_mask, (21,21), 5)
    
    ### Contour Extraction ---------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Find contours of mask:
    contours, hierarchy = cv2.findContours(mask_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_contours = cv2.cvtColor(segmentation_mask.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask_contours, contours, -1, (0,255,0), 2)

    ### Bounding Box Extraction ----------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Find resistor bounding boxes using contours:
    mask_boundings = mask_contours.copy()
    bounding_rects = []                                                         #Init bounding box return array
    for cnt in contours:                                                        #Iterate through contours:
        rect = cv2.minAreaRect(cnt)                                                 #Find minimum area rectangle
        box = cv2.boxPoints(rect)                                                   #Find box points from rectangle
        box = np.intp(box)                      
    
        dimensions = rect[1]                                                        #Enforce min resolution to cull stray noise
        if (min(dimensions) < min_resistor_res): continue
    
        cv2.drawContours(mask_boundings,[box],0,(0,0,255),3)                        #Draw bounding box on mask
        bounding_rects.append(rect)                                                 #Append bounding box to return array

    ### Resistor Extraction --------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Extract resistors using bounding boxes:
    extracted_masks = []                                                        #Init extracted masks return array     
    extracted_resistors = []                                                    #Init extracted resistors return array
    for rect in bounding_rects:                                                 #Iterate through bounding boxes:
        
        mask = func.extractImgUsingRect(segmentation_mask.copy(),                   #Extract mask using bounding box
                                        rect,
                                        target_width = target_resistor_width)
        
        resistor = func.extractImgUsingRect(filtered_input_image.copy(),            #Extract resistor using bounding box
                                            rect, 
                                            target_width = target_resistor_width)
    
        extracted_masks.append(mask)                                                #Append mask to return array
        extracted_resistors.append(resistor)                                        #Append resistor to return array
    

    return (extracted_masks, extracted_resistors)



#################################################################################################################################################################################################
### Refine Resistor Masks #######################################################################################################################################################################
# ╰┈➤ Refine individual extracted resistor masks using morphological operations to patch holes and (hopefully) form a contiguous complete mask:        
    
def Refine_Resistor_Mask(mask):
    
    ### Mask Shape Extraction -------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Extract mask dimensions and midpoints:
    h, w = mask.shape
    mid_h = int(h/2)
    mid_w = int(w/2)
    
    ### Mask Joining ----------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Find each horizontal edge of the mask and join them a contiguous line to assist coming morphological operations:
    x1, x2 = -1, -1
    for i in range(w):                                                          #Iterate through mask width:
        if (mask[mid_h, i] > 200) and (x1 == -1):                                   #If first left-most white pixel found, set x1
            x1 = i
        if (mask[mid_h, w-i-1] > 200) and (x2 == -1):                               #If first right-most white pixel found, set x2
            x2 = w-i-1
        if (x1 != -1 and x2 != -1):                                                 #If both x1 and x2 set, break loop
            continue
    mask[mid_h-5:mid_h+5, x1:x2] = 255                                          #Join the horizontal edges of the mask
        
    ### Mask Dilation ----------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Dilate mask to fill missing vertical slices. Use custom kernels to target horizontal expansion and prevent vertical expansion:

    dilation_kernel_up = np.array([[1, 1, 1, 1, 1],                             #Custom kernel for upper half dilation.
                                   [1, 1, 1, 1, 1],                             # - 1's only in the upper half to prevent vertical expansion
                                   [1, 1, 1, 1, 1],                             # - Will not consider pixels below operating point
                                   [0, 0, 0, 0, 0],                             # - Hence will not expand mask vertically upwards
                                   [0, 0, 0, 0, 0]], dtype="uint8")             # - Will only expand sideways and downwards
                                                                                # - Allows for aggressive dilation without losing resistor shape

    dilation_kernel_down = np.roll(dilation_kernel_up, 2, axis=0)               #Flip the upper half kernel to create lower half kernel

    dilated = mask.copy()
    dilated[0:mid_h, :] = cv2.dilate(mask[0:mid_h, :], dilation_kernel_up, iterations = 10)
    dilated[mid_h:h-1, :] = cv2.dilate(mask[mid_h:h-1, :], dilation_kernel_down, iterations = 10)

    ### Mask Erosion -----------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Erode mask to remove noise and shave off any excess expansion from dilation:
    erosion_dist = 2
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_dist + 1, 2 * erosion_dist + 1))
    erosion_kernel_flat = np.ones((1, 2 * erosion_dist + 1))

    eroded = cv2.erode(dilated, erosion_kernel, iterations = 2)
    eroded = cv2.erode(eroded, erosion_kernel_flat, iterations = 10)
    #eroded = cv2.erode(eroded, erosion_kernel, iterations = 4)
 
    ### Mask Closing -----------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Close mask to fill any remaining gaps. Unlike the dilation, this only targets holes, not missing slices:
    closing_dist = 3
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * closing_dist + 1, 2 * closing_dist + 1))

    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, closing_kernel, iterations = 2)
    

    return closed



#################################################################################################################################################################################################
### Inner Resistor Extraction ###################################################################################################################################################################
# ╰┈➤ Extract the inner portion of the resistor containing the bands and body:

def Inner_Resistor_Extraction(resistor, mask, target_resistor_width):

    ### Mask Shape Extraction --------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Extract mask dimensions and midpoints:
    h, w = mask.shape
    mid_h = int(h/2)
    mid_w = int(w/2)
    
    ### Inner Resistor Extraction ----------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Cut out the inner portion of the resistor such that only the bands and body are contained:
    
    rng = int(target_resistor_width/10)                                         #Horizontal range of white pixels required to qualify as start of resistor body

    y1, y2 = -1, -1                                                         ######### Vertical bounds: -----
    for i in range(mid_h-1):                                                    #Iterate through half the mask height:
        if (np.all(mask[i, mid_w-rng:mid_w+rng] == 255)) and (y1 == -1):            #If all central pixels in the horizontal range are white, set y1 (upper bounds)
            y1 = i
        if (np.all(mask[h-i-1, mid_w-rng:mid_w+rng] == 255)) and (y2 == -1):        #If all central pixels in the horizontal range are white, set y2 (lower bounds)
            y2 = h-i-1
        if (y1 != -1 and y2 != -1):                                                 #If both y1 and y2 set, break loop
            continue
    if (y1 == -1 or y2 == -1): return None                                      #If either y1 or y2 not set, assume invalid resistor extraction and return None
    
    h_reduce = 0.4                                                              #Reduce vertical bounds by selected factor (40%) for safety margin and to avoid shadows
    h_inner = y2 - y1
    if (h_reduce > 0.0):
        y1 = int(y1 + ((h_inner * h_reduce)/2))
        y2 = int(y2 - ((h_inner * h_reduce)/2))
        h_inner = y2 - y1
        
    if (h_inner < 14): return None                                              #If vertical bounds are too small, assume invalid resistor extraction and return None
    
    x1, x2 = -1, -1                                                         ######### Horizontal bounds: -----
    for i in range(mid_w-1):                                                    #Iterate through half the mask width:
        if (np.all(mask[y1:y2, i] == 255)) and (x1 == -1):                          #If all pixels in the found vertical bounds are white, set x1 (left bounds)
            x1 = i
        if (np.all(mask[y1:y2, w-i-1] == 255)) and (x2 == -1):                      #If all pixels in the found vertical bounds are white, set x2 (right bounds)
            x2 = w-i-1
        if (x1 != -1 and x2 != -1):                                                 #If both x1 and x2 set, break loop
            continue
    if (x1 == -1 or x2 == -1): return None                                      #If either x1 or x2 not set, assume invalid resistor extraction and return None
    
    w_reduce = 0.0                                                              #Reduce horizontal bounds by selected factor (0%) for safety margin and to avoid shadows
    w_inner = x2 - x1
    if (w_reduce > 0.0):   
        x1 = int(x1 + ((w_inner * w_reduce)/2))
        x2 = int(x2 - ((w_inner * w_reduce)/2))
        w_inner = x2 - x1
    
    inner_resistor = resistor[y1:y2, x1:x2]                                     #Extract inner resistor using calculated bounds
    
    return inner_resistor
  


#################################################################################################################################################################################################
### Band Segmentation ###########################################################################################################################################################################     
# ╰┈➤ Segment the resistor bands from the resistor body: 

def Band_Segmentation(inner_resistor):   

    ### Mask Shape Extraction --------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Extract inner dimensions and midpoints:
    h, w, _ = inner_resistor.shape
    mid_h = int(h/2)
    mid_w = int(w/2)
    
    ### Body Colour Extraction -------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Extract resistor body colour as most common/dominant K-Means cluster:
 
    k_clusters = 5                      #Number of K-Means clusters to use. Using 5 allows for band colours, bright spots, and shadows to not interfere with body colour
    v_min = 60                          #Minimum V-Value (in HSV) for dominant colour to be considered valid (will use second most common colour if not)
    
    img_bodycolour = inner_resistor.copy()
    img_bodycolour[:] = func.ExtractDominantColour(inner_resistor[mid_h-7:mid_h+7, :], 
                                                   k_clusters=5, v_min=60)

    ### Body Colour Subtraction ------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Find absolute difference between inner resistor and body colour to isolate bands:
    abs_diff = cv2.absdiff(inner_resistor, img_bodycolour)

    ### OTSU Threshold ---------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Automatic threshold on the absdiff image using HSV 'V' Channel to segment bands:

    diff_HSV = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2HSV)                        #Convert absdiff image to HSV
    diff_V = diff_HSV[:, :, 2]                                                  #Extract V-Channel
    otsu_threshold, _ = cv2.threshold(diff_V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  #Calculate OTSU threshold
    _, band_mask = cv2.threshold(diff_V, int(otsu_threshold * 0.9), 255, cv2.THRESH_BINARY)      #Apply leanient OTSU threshold (reduced by 0.9 factor) to V-Channel
       
    ### Band Mask Dilation and Erosion -----------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Dilate and erode resulting band mask to patch holes and form contiguous bands:
    dilation_dist = 2
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_dist + 1, 2 * dilation_dist + 1))
    band_mask = cv2.dilate(band_mask, dilation_kernel, iterations = 1)
    #test dilating with column kernel instead

    erosion_dist = 1
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_dist + 1, 2 * erosion_dist + 1))
    band_mask = cv2.erode(band_mask, erosion_kernel, iterations = 3)

    ### Vertical Slice Enforcement ----------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Only keep purely white vertical columns (with some leeway), correlating to bands: 
    
    leeway = 5                                                                  #Top/bottom leeway to not be considered when assessing columns
    band_mask_final = band_mask.copy()
    
    for i in range(w):                                                          #Iterate through mask width:
        if (np.any(band_mask[leeway:h-leeway, i] == 0)):                           
            band_mask_final[0:h, i] = 0
        else:
            band_mask_final[0:h, i] = 255

        
    return (band_mask_final, abs_diff)



#################################################################################################################################################################################################
### Band Extraction #############################################################################################################################################################################     
# ╰┈➤ Extract the individual bands from inner resistor cutout using the band mask:

def Band_Extraction(mask, inner_resistor, bodycolour_abs_diff):
    
    ### Mask Shape Extraction --------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Extract mask dimensions and midpoints:
    h, w = mask.shape
    mid_h = int(h/2)
    mid_w = int(w/2)
    
    ### Band Info Extraction ---------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Extract individual band start/endpoints, and widths, as provided by the band mask:
    
    min_width = 3                       #Minimum width required for a band to be considered valid    
    
    extracted_bands_info = []                                                   #Init extracted bands info return array
    bBand = False                                                               #Init bool used to keep track of band presence
    x0 = 0
    for j in range(w):                                                          #Iterate through mask width:
        
        if (mask[mid_h, j] == 255) and (bBand == False):                            #If white pixel found and no band presence currently tracked:
            bBand = True                                                                #Start tracking band presence
            x0 = j                                                                      #Set band start point
            
        if (mask[mid_h, j] == 0) and (bBand == True):                               #If black pixel found and band presence currently tracked:
            bBand = False                                                               #Stop tracking band presence
            band_info = [x0, j, j-x0]                                                   #Save band info (start, end, width)
            if band_info[2] > min_width:                                                #If band width is above minimum threshold, append to return array
                extracted_bands_info.append(band_info)
                
        if (j == w-1) and (bBand == True):                                          #If end of mask reached and band presence currently tracked:
            band_info = [x0, w, w-x0]                                                   #Save band info (start, end, width)
            if band_info[2] > min_width:                                                #If band width is above minimum threshold, append to return array
                extracted_bands_info.append(band_info)
    
    if (len(extracted_bands_info) < 3): return None                             #If less than three bands found, assume invalid extraction and return None
                                                                                        # (Too few to form a valid resistor or extrapolate missing bands)
                    
    ### Extrapolate Missing Bands ----------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Analyse band widths and seperations to extrapolate the locations of missing bands and search for their presense              

        
    #If there are less than 5 bands, or there are 5 bands with one touching the image edge (potential false positive):
    if (len(bands_info) < 5) or (len(bands_info) == 5 and (bands_info[0][0] == 0 or bands_info[-1][1] == w)):
            
        band_seperations = []
        band_widths = []
        for j in range(len(bands_info) - 1):
            band_seperations.append(bands_info[j+1][0] - bands_info[j][1])
            band_widths.append(bands_info[j][2])
        band_widths.append(bands_info[-1][2])
        
        #Find the largest seperation between bands
        largest_separation_index = np.argmax(band_seperations)
        largest_separation = band_seperations[largest_separation_index]
        
        #Find the average band seperation EXCLUDING the largest seperation
        typical_seperations = band_seperations.copy()
        typical_seperations.pop(largest_separation_index)
        typical_seperation = np.mean(typical_seperations)

        ave_band_width = np.mean(band_widths)           #Find the average band width
        target_band_width = int(0.8 * ave_band_width)   #Find the target band width as 0.75 * average band width
                    
        start_seperation = bands_info[0][0]     #Find the gap between the image start and the first band
        end_seperation = w - bands_info[-1][1]  #Find the gap between the image end and the last band
        
        #If the gap between the start of image and first band is sufficiently large, search for a missing band:
        if (start_seperation - target_band_width >= 1.35 * typical_seperation):
            #Find the middle point between the image start and the first band
            xm = 0 + int(start_seperation/2)
            
            print("\nIndex: {} - Assuming missing band near start, searching...".format(i, xm))
                
            #Search for the missing band
            abs_diff = extracted_resistors_inner_difference[i]
            missing_band, continuity = func.LocateMissingBand(xm, start_seperation/2, target_band_width, abs_diff)
                
            #Given sufficient continuity, insert a new band
            if (continuity >= 0.8):
                bands_info.insert(0, missing_band)
                print("... Band located with continuity: {:.3f} -> Inserting band at: {}".format(continuity, missing_band[0]+int(missing_band[2]/2)))

                new_seperation = bands_info[1][0] - bands_info[0][1]
                if (new_seperation > largest_separation):
                    largest_separation = new_seperation
                    largest_separation_index = 0
                else: largest_separation_index += 1
                    
            else: print("... Band not located - Continuity too low: {:.3f}".format(continuity))
            
        #If the gap between the last band and end of image is sufficiently large, search for a missing band:
        if (end_seperation - target_band_width >= 1.35 * typical_seperation):
            #Find the middle point between the last band and end of image
            xm = bands_info[-1][1] + int(end_seperation/2)
                
            print("\nIndex: {} - Assuming missing band near end, searching...".format(i, xm))
                
            #Search for the missing band
            abs_diff = extracted_resistors_inner_difference[i]
            missing_band, continuity = func.LocateMissingBand(xm, end_seperation/2, target_band_width, abs_diff)
                
            #Given sufficient continuity, insert a new band
            if (continuity >= 0.8):
                bands_info.append(missing_band)
                print("... Band located with continuity: {:.3f} -> Inserting band at: {}".format(continuity, missing_band[0]+int(missing_band[2]/2)))
            else: print("... Band not located - Continuity too low: {:.3f}".format(continuity))
            
        #If the largest seperation is sufficiently large, search for a missing band:
        if (largest_separation - target_band_width >= 1.35 * typical_seperation):
            #Find the middle point between the two bands
            xm = bands_info[largest_separation_index][1] + int(largest_separation/2)
             
            print("\nIndex: {} - Assuming missing band near {}, searching...".format(i, xm))
                
            #Search for the missing band
            abs_diff = extracted_resistors_inner_difference[i]
            missing_band, continuity = func.LocateMissingBand(xm, largest_separation/2, target_band_width, abs_diff)
                
            #Given sufficient continuity, insert a new band
            if (continuity >= 0.7):
                bands_info.insert(largest_separation_index+1, missing_band)
                print("... Band located with continuity: {:.3f} -> Inserting band at: {}".format(continuity, missing_band[0]+int(missing_band[2]/2)))
            else: 
                print("... Band not located - Continuity too low: {:.3f}".format(continuity))
                    
                #If there are still less than 5 bands, give more leeway to insert a new band
                if (len(bands_info) < 5) and (continuity >= 0.2):
                    x0 = xm - int(target_band_width/2)
                    x1 = xm + int(target_band_width/2)
                    bands_info.insert(largest_separation_index+1, [x0, x1, target_band_width])
                    print("... Still <5 bands -> Inserting band in middle at: {}".format(x0+int(target_band_width/2)))
                        
        #If there are still only 3 bands (spacing is uniform if this is still true), search for a missing band in each gap:
        elif (len(bands_info) == 3):
            k = 0
            for j in range(len(bands_info) - 1):
                #Find the gap between the bands
                seperation = bands_info[j+k+1][0] - bands_info[j+k][1]
  
                #Find the middle point between the two bands
                xm = bands_info[j+k][1] + int(seperation/2)
             
                print("\nIndex: {} - Three uniformly seperated bands present, searching...".format(i, xm))

                #Search for the missing band
                abs_diff = extracted_resistors_inner_difference[i]
                missing_band, continuity = func.LocateMissingBand(xm, seperation/2, target_band_width, abs_diff)
                
                #Given sufficient continuity, insert a new band
                if (continuity >= 0.55):
                    bands_info.insert(j+k+1, missing_band)
                    k = 1
                    print("... Band located with continuity: {:.3f} -> Inserting band at: {}".format(continuity, missing_band[0]+int(missing_band[2]/2)))
                else: 
                    print("... Band not located - Continuity too low: {:.3f}".format(continuity))
                       
            
    #If there are now more than 5 bands, remove any that are touching the image edge (likely false positives) 
    if (len(bands_info) > 5):
        if (bands_info[0][0] == 0):
            bands_info.pop(0)
        elif (bands_info[-1][1] == w):
            bands_info.pop(-1)
    #extracted_bands_info.append(bands_info)
        
    #cv2.waitKey(0)

    ##############################################################
    ### Extract bands using obtained band info
    extracted_bands = []
    for i in range(len(extracted_bands_info)):
        bands_info = extracted_bands_info[i]
        resistor_inner = extracted_resistors_inner[i] #THIS WONT ALWAYS WORK!!!! Need to resolve mismatching indexes (kinda fixed)
    
        preview = resistor_inner.copy()
        preview[:] = (0, 0, 0)
    
        bands = []
        for band_info in bands_info:
            band = resistor_inner[:, band_info[0]:band_info[1]]
        
            preview[:, band_info[0]:band_info[1]] = band
        
            band_resize = cv2.resize(band, target_band_shape)
            bands.append(band_resize)
        extracted_bands.append(bands)
    
        cv2.imshow(str(i) + "c", preview)
    #cv2.waitKey(0)


    
def Band_Colour_Classification_CNN():

def Decode_Resistance():


def Export_Bands_For_Training(extracted_bands, target_shape, target_dest, img_index):
    
    ################################################
    # Export the bands for training
    for i in range(len(extracted_bands)):
        for j in range(len(extracted_bands[i])):
            band = extracted_bands[i][j]
            band_resize = cv2.resize(band, target_shape)
            cv2.imwrite('data_training/' + target_dest + '/' + str(img_index) + '_' + str(i) + '_' + str(j) + '.jpg', band_resize)
            print("Exporting band: " + 'data_training/' + target_dest + '/' + str(img_index) + '_' + str(i) + '_' + str(j) + '.jpg')
    ################################################)
   
def Classify_Band_Colours_CNN(extracted_bands, CNN):
    extracted_bands_classes = []
    test = 0    
    for bands in extracted_bands:
        band_classes = []
        for band in bands:
            band_RGB = cv2.cvtColor(band, cv2.COLOR_BGR2RGB)    # CNN is trained using RGB images
            band_RGB = band_RGB / 255.0                         # CNN is trained using images with pixel values between 0 and 1
            band_RGB = band_RGB.reshape(1, 30, 12, 3)                   # CNN expects batch size, height, width, channels
            band_predictions = CNN.predict(band_RGB, verbose=0)                # Use the CNN model to predict the class of the band
            band_class = np.argmax(band_predictions)            # Classify as the class with the highest probability
            band_classes.append(band_class)
        extracted_bands_classes.append(band_classes)
        print(band_classes)
    return extracted_bands_classes

def DecodeResistance(extracted_bands_classes, tolerance_lookup):
    decoded_resistance_num = []
    decoded_resistance_string = []
    decoded_tolerance = []
    for band_classes in extracted_bands_classes:
        if (len(band_classes) != 5): continue
    
        band_classes = np.array(band_classes)
    
        bInvertReadDirection = False
        if ((band_classes[0] == 0)
         or (np.any(band_classes[0:3] > 9))
         or (tolerance_lookup[band_classes[4]] == "ERR%")):
            bInvertReadDirection = True
    
        resistance_num = -1.0
        num = -1.0
        tolerance = "ERR%"
        if (bInvertReadDirection == False):
            num = int(str(band_classes[0]) + str(band_classes[1]) + str(band_classes[2]))
            mult = band_classes[3]
            tolerance = tolerance_lookup[band_classes[4]]
        elif (bInvertReadDirection == True):
            num = int(str(band_classes[4]) + str(band_classes[3]) + str(band_classes[2]))
            mult = band_classes[1]
            tolerance = tolerance_lookup[band_classes[0]]   
        if (mult > 9): mult = 0 - (mult-9)
        resistance_num = num * (10.0 ** mult)
    
        resistance_string = "ERR"
        resistance_string = func.int_to_metric_string(resistance_num)
    
        decoded_resistance_num.append(resistance_num)
        decoded_resistance_string.append(resistance_string)
        decoded_tolerance.append(tolerance)
    
        print(resistance_string + '\u03A9')
    cv2.waitKey(0) 