import cv2
import numpy as np

import Functions as func

##################################################################################################################################################################################################
### Input Image Preprocessing ####################################################################################################################################################################
# ╰┈➤ Preprocess the input image to resize and filter out noise:

def Input_Image_Preprocessing(input_image, max_input_res):
    
    ### Resize Image ---------------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Resize input image to reduce processing time and memory usage, and allow for more consistent morphological operations:
    h, w, _ = input_image.shape
    if (h > max_input_res[0] or w > max_input_res[1]):
        
        h_scale = h / max_input_res[0]
        w_scale = w / max_input_res[1]
    
        scale = max(h_scale, w_scale)
        new_dim = [int(w / scale), int(h / scale)]
        input_image = cv2.resize(input_image, new_dim)

    ### Median Filter --------------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ To remove noise but preserve edges
    filtered_input_image = cv2.bilateralFilter(input_image,25,25,25)

    return filtered_input_image

#################################################################################################################################################################################################
### Resistor Segmentation #######################################################################################################################################################################
# ╰┈➤ Isolate the resistors from the background:

def Resistor_Segmentation(filtered_input_image, gamma_correction=3, threshold_factor=1.0):

    ### Gamma Correct --------------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Increase gamma to drown out background saturation:
    image_gamma_correct =  func.applyGammaCorrection(filtered_input_image, gamma_correction) #consider opposite for dark background?

    ### OTSU Threshold -------------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Automatic threshold on HSV 'S' Channel to isolate coloured (saturated) resistor body from background:
    image_HSV = cv2.cvtColor(image_gamma_correct, cv2.COLOR_BGR2HSV)
    otsu_threshold, mask_OTSU = cv2.threshold(image_HSV[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
    if (threshold_factor != 1.0): _, mask_OTSU = cv2.threshold(image_HSV[:, :, 1], int(otsu_threshold * threshold_factor), 255, cv2.THRESH_BINARY)

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

def Resistor_Extraction(segmentation_mask, preprocessed_input_image, min_resistor_res, target_resistor_width):
    
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
        bounding_rects.append(rect) 

    ### Resistor Extraction --------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Extract resistors using bounding boxes:
    extracted_masks = []                                                        #Init extracted masks return array     
    extracted_resistors = []                                                    #Init extracted resistors return array
    for rect in bounding_rects:                                                 #Iterate through bounding boxes:
        
        mask = func.extractImgUsingRect(segmentation_mask.copy(),                   #Extract mask using bounding box
                                        rect,
                                        target_width = target_resistor_width)
        
        resistor = func.extractImgUsingRect(preprocessed_input_image.copy(),        #Extract resistor using bounding box
                                            rect, 
                                            target_width = target_resistor_width)
    
        extracted_masks.append(mask)                                                #Append mask to return array
        extracted_resistors.append(resistor)                                        #Append resistor to return array
    

    return (extracted_masks, extracted_resistors, mask_boundings, bounding_rects)



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
    dilated[0:mid_h, :] = cv2.dilate(mask[0:mid_h, :], dilation_kernel_up, iterations = 6)
    dilated[mid_h:h-1, :] = cv2.dilate(mask[mid_h:h-1, :], dilation_kernel_down, iterations = 6)

    ### Mask Erosion -----------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Erode mask to remove noise and shave off any excess expansion from dilation:
    erosion_dist = 2
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_dist + 1, 2 * erosion_dist + 1))
    erosion_kernel_flat = np.ones((1, 2 * erosion_dist + 1))

    eroded = cv2.erode(dilated, erosion_kernel, iterations = 2)
    eroded = cv2.erode(eroded, erosion_kernel_flat, iterations = 6)
    #eroded = cv2.erode(eroded, erosion_kernel, iterations = 4)
 
    ### Mask Closing -----------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Close mask to fill any remaining gaps. Unlike the dilation, this only targets holes, not missing slices:
    closing_dist = 4
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

    ### Vertical Bounds --------------------------------------------------------#----------------------------#
    y1, y2 = -1, -1                                                         
    for i in range(mid_h-1):                                                    #Iterate through half the mask height:
        if (np.all(mask[i, mid_w-rng:mid_w+rng] == 255)) and (y1 == -1):            #If all central pixels in the horizontal range are white, set y1 (upper bounds)
            y1 = i
        if (np.all(mask[h-i-1, mid_w-rng:mid_w+rng] == 255)) and (y2 == -1):        #If all central pixels in the horizontal range are white, set y2 (lower bounds)
            y2 = h-i-1
        if (y1 != -1 and y2 != -1):                                                 #If both y1 and y2 set, break loop
            continue
    if (y1 == -1 or y2 == -1): return (None, None)                                  #If either y1 or y2 not set, assume invalid resistor extraction and return None
    
    h_reduce = 0.4                                                              #Reduce vertical bounds by selected factor (40%) for safety margin and to avoid shadows
    h_inner = y2 - y1
    if (h_reduce > 0.0):
        y1 = int(y1 + ((h_inner * h_reduce)/2))
        y2 = int(y2 - ((h_inner * h_reduce)/2))
        h_inner = y2 - y1
        
    if (h_inner < 14): return (None, None)                                      #If vertical bounds are too small, assume invalid resistor extraction and return None
    
    ### Horizontal Bounds ------------------------------------------------------#----------------------------#
    x1, x2 = -1, -1                                                         
    for i in range(mid_w-1):                                                    #Iterate through half the mask width:
        if (np.all(mask[y1:y2, i] == 255)) and (x1 == -1):                          #If all pixels in the found vertical bounds are white, set x1 (left bounds)
            x1 = i
        if (np.all(mask[y1:y2, w-i-1] == 255)) and (x2 == -1):                      #If all pixels in the found vertical bounds are white, set x2 (right bounds)
            x2 = w-i-1
        if (x1 != -1 and x2 != -1):                                                 #If both x1 and x2 set, break loop
            continue
    if (x1 == -1 or x2 == -1): return (None, None)                              #If either x1 or x2 not set, assume invalid resistor extraction and return None
    
    w_reduce = 0.0                                                              #Reduce horizontal bounds by selected factor (0%) for safety margin and to avoid shadows
    w_inner = x2 - x1
    if (w_reduce > 0.0):   
        x1 = int(x1 + ((w_inner * w_reduce)/2))
        x2 = int(x2 - ((w_inner * w_reduce)/2))
        w_inner = x2 - x1
    
    inner_resistor = resistor[y1:y2, x1:x2]                                     #Extract inner resistor using calculated bounds

    return (inner_resistor, x1)
  


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

def Band_Extraction(mask, inner_resistor, bodycolour_abs_diff, target_band_shape, object_index):
    
    ### Mask Shape Extraction --------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Extract mask dimensions and midpoints:
    h, w = mask.shape
    mid_h = int(h/2)
    mid_w = int(w/2)
    
    ### Band Info Extraction ---------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Extract individual band start/endpoints, and widths, as provided by the band mask:
    
    min_width = 3                       #Minimum width required for a band to be considered valid    
    
    bands_info = []                                                             #Init extracted bands info return array
    bBand = False                                                               #Init bool used to keep track of band presence
    x0 = 0
    for i in range(w):                                                          #Iterate through mask width:
        
        if (mask[mid_h, i] == 255) and (bBand == False):                            #If white pixel found and no band presence currently tracked:
            bBand = True                                                                #Start tracking band presence
            x0 = i                                                                      #Set band start point
            
        if (mask[mid_h, i] == 0) and (bBand == True):                               #If black pixel found and band presence currently tracked:
            bBand = False                                                               #Stop tracking band presence
            band_info = [x0, i, i-x0]                                                   #Save band info (start, end, width)
            if band_info[2] > min_width:                                                #If band width is above minimum threshold, append to return array
                bands_info.append(band_info)
                
        if (i == w-1) and (bBand == True):                                          #If end of mask reached and band presence currently tracked:
            band_info = [x0, w, w-x0]                                                   #Save band info (start, end, width)
            if band_info[2] > min_width:                                                #If band width is above minimum threshold, append to return array
                bands_info.append(band_info)
    
    if (len(bands_info) < 3): return (None, None, None)                             #If less than three bands found, assume invalid extraction and return None
                                                                                    # (Too few to form a valid resistor or extrapolate missing bands)
                    
    ### Extrapolate Missing Bands ----------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Analyse band widths and separations to extrapolate the locations of missing bands and search for their presense              
    
    if ((len(bands_info) < 5)                                                   #If there are less than 5 bands, or there are 5 bands with one touching the image edge (potential false positive):
     or (len(bands_info) == 5 and (bands_info[0][0] == 0 or bands_info[-1][1] == w))):
            
        band_separations = []                                                       #Init band separations array
        band_widths = []                                                            #Init band widths array
        for i in range(len(bands_info) - 1):                                        #Iterate through bands (excl. last band):
            band_separations.append(bands_info[i+1][0] - bands_info[i][1])              #Save the separation between the band and the next band
            band_widths.append(bands_info[i][2])                                        #Save the width of the band
        band_widths.append(bands_info[-1][2])                                       #Save the width of the last band
        
        largest_separation_index = np.argmax(band_separations)                      #Find the index of the largest separation between bands
        largest_separation = band_separations[largest_separation_index]             #Find the largest separation between bands
        
        typical_separations = band_separations.copy()                               
        typical_separations.pop(largest_separation_index)                           #Build an array of band separations EXCLUDING the largest separation
        typical_separation = np.mean(typical_separations)                           #Find the typical band separation EXCLUDING the largest separation

        ave_band_width = np.mean(band_widths)                                       #Find the average band width
        target_band_width = int(0.8 * ave_band_width)                               #Find the target band width as 0.75 * average band width
                    
        start_separation = bands_info[0][0]                                         #Find the gap between the image start and the first band
        end_separation = w - bands_info[-1][1]                                      #Find the gap between the image end and the last band
        
        ### Start of Image Search --------------------------------------------------#----------------------------#
        if (start_separation - target_band_width >= 1.35 * typical_separation):     #If the gap between the start of image and first band is sufficiently large, search for a missing band:
            xm = 0 + int(start_separation/2)                                            #Find the middle point between the image start and the first band
            print("\nIndex: {} - Assuming missing band near start, searching...".format(object_index, xm))
                                                                                        
            missing_band, continuity = func.LocateMissingBand(xm, start_separation/2,   #Search for the missing band around xm with a search range of half the gap
                                                              target_band_width, bodycolour_abs_diff)
                
            if (continuity >= 0.8):                                                     
                bands_info.insert(0, missing_band)                                          #Insert the missing band at the start of the bands info array
                print("... Band located with continuity: {:.3f} -> Inserting band at: {}".format(continuity, missing_band[0]+int(missing_band[2]/2)))

                new_separation = bands_info[1][0] - bands_info[0][1]                        #Find the new separation between the first two bands
                if (new_separation > largest_separation):                                   #If the new separation is larger than the previous largest separation:
                    largest_separation = new_separation                                         #Update the largest separation
                    largest_separation_index = 0                                                #Update the index of the largest separation
                else: largest_separation_index += 1                                         #Otherwise, nudge the index of the largest separation by 1
                    
            else: print("... Band not located - Continuity too low: {:.3f}".format(continuity))
        
        ### End of Image Search ----------------------------------------------------#----------------------------#
        if (end_separation - target_band_width >= 1.35 * typical_separation):       #If the gap between the last band and end of image is sufficiently large, search for a missing band:
            xm = bands_info[-1][1] + int(end_separation/2)                              #Find the middle point between the last band and end of image
            print("\nIndex: {} - Assuming missing band near end, searching...".format(object_index, xm))
                                                                                        
            missing_band, continuity = func.LocateMissingBand(xm, end_separation/2,     #Search for the missing band around xm with a search range of half the gap
                                                              target_band_width, bodycolour_abs_diff)
                
            if (continuity >= 0.8):                                                     #Given the located band is of sufficient continuity:
                bands_info.append(missing_band)                                             #Append the missing band to the end of the bands info array
                print("... Band located with continuity: {:.3f} -> Inserting band at: {}".format(continuity, missing_band[0]+int(missing_band[2]/2)))
            else: print("... Band not located - Continuity too low: {:.3f}".format(continuity))
        
        ### Largest separation Search ----------------------------------------------#----------------------------#
        if (largest_separation - target_band_width >= 1.35 * typical_separation):   #If the largest separation is sufficiently large, search for a missing band:
            xm = bands_info[largest_separation_index][1] + int(largest_separation/2)    #Find the middle point between the two bands
            print("\nIndex: {} - Assuming missing band near {}, searching...".format(object_index, xm))
                                                                                        
            missing_band, continuity = func.LocateMissingBand(xm, largest_separation/2, #Search for the missing band around xm with a search range of half the gap
                                                              target_band_width, bodycolour_abs_diff)

            if (continuity >= 0.7):                                                     #Given the located band is of sufficient continuity:
                bands_info.insert(largest_separation_index+1, missing_band)                 #Insert the missing band after the index of the largest separation
                print("... Band located with continuity: {:.3f} -> Inserting band at: {}".format(continuity, missing_band[0]+int(missing_band[2]/2)))
            else: 
                print("... Band not located - Continuity too low: {:.3f}".format(continuity))
                    
                if (((len(bands_info) < 5) and (continuity >= 0.2))                     #Otherwise, if there are still less than 5 bands, give more leeway to insert a new band
                 or ((len(bands_info) == 5) and (bands_info[0][0] == 0 or bands_info[-1][1] == w) and (continuity >= 0.4))):
                    x0 = xm - int(target_band_width/2)                                      
                    x1 = xm + int(target_band_width/2)                                      #Insert a band directly in the middle of the largest separation (rather than at search location)
                    bands_info.insert(largest_separation_index+1, [x0, x1, target_band_width])
                    print("... Still <5 bands -> Inserting band in middle at: {}".format(x0+int(target_band_width/2)))
        
        ### Uniform Spacing Search -------------------------------------------------#----------------------------#
        elif (len(bands_info) == 3):                                                #If there are still only 3 bands (spacing is uniform if this is still true), search for a missing band in each gap:
            k = 0
            for i in range(len(bands_info) - 1):                                        #Iterate through bands (excl. last band):
                separation = bands_info[i+k+1][0] - bands_info[i+k][1]                      #Find the gap between the bands
                xm = bands_info[i+k][1] + int(separation/2)                                 #Find the middle point between the two bands
                print("\nIndex: {} - Three uniformly seperated bands present, searching...".format(object_index, xm))
                                                                                            
                missing_band, continuity = func.LocateMissingBand(xm, separation/2,         #Search for the missing band around xm with a search range of half the gap
                                                                  target_band_width, bodycolour_abs_diff)
                
                if (continuity >= 0.55):                                                    #Given the located band is of sufficient continuity:
                    bands_info.insert(i+k+1, missing_band)                                      #Insert the missing band after the index of the largest separation
                    k = 1                                                                       #Increment k to account for the inserted band in the next iteration
                    print("... Band located with continuity: {:.3f} -> Inserting band at: {}".format(continuity, missing_band[0]+int(missing_band[2]/2)))
                else: 
                    print("... Band not located - Continuity too low: {:.3f}".format(continuity))               
                    
    ### Remove False Positives -------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ If the extrapolation or extraction process has resulted in more than 5 bands, remove assumed false positives  
                    
    if (len(bands_info) > 5):                                                           #If there are more than 5 bands:
        if (bands_info[0][0] == 0):                                                         #Remove any band touching the image start (likely false positives) 
            bands_info.pop(0)
        elif (bands_info[-1][1] == w):                                                      #Otherwise, remove any band touching the image end (likely false positives) 
            bands_info.pop(-1)
            
    ### Find Largest Band Separation -------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Find the index of the largest separation between bands (used to determine the reading direction):
            
    band_separations = []                                                       #Init band separations array
    for j in range(len(bands_info) - 1):                                        #Iterate through bands (excl. last band):
        band_separations.append(bands_info[j+1][0] - bands_info[j][1])              #Save the separation between the band and the next band
        
    largest_separation_index = np.argmax(band_separations)                      #Find the index of the largest separation between bands
       
    ### Band Image Extraction --------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Using the obtained band info, extract the individual bands from the inner resistor cutout:
    
    preview = inner_resistor.copy()                                                     #Prepare preview image, same size as inner resistor
    preview[:] = (0, 0, 0)                                                              #Make preview image black
        
    extracted_bands = []                                                                #Init extracted bands return array
    for band_info in bands_info:                                                        #Iterate through obtained band info:
        band = inner_resistor[:, band_info[0]:band_info[1]]                                 #Cut out the band from the inner resistor image using the band info
        
        preview[:, band_info[0]:band_info[1]] = band                                        #Overlay the extracted band on the preview image
        
        band_resize = cv2.resize(band, target_band_shape)                                   #Resize the band to the target shape for CNN classification
        extracted_bands.append(band_resize)                                                 #Append the resized band to the return 
    

    return (extracted_bands, preview, largest_separation_index)



#################################################################################################################################################################################################
### Band Colour Classification ##################################################################################################################################################################     
# ╰┈➤ Classify the colours of the extracted bands using the trained CNN model:
    
def Band_Colour_Classification_CNN(extracted_bands, CNN):

    extracted_band_classes = []                                                         #Init extracted bands classes return array
    for band in extracted_bands:                                                        #Iterate through extracted bands:
        band_RGB = cv2.cvtColor(band, cv2.COLOR_BGR2RGB)                                    #CNN is trained using RGB images
        band_RGB = band_RGB / 255.0                                                         #CNN is trained using images with pixel values between 0 and 1
        band_RGB = band_RGB.reshape(1, 30, 12, 3)                                           #CNN expects batch size, height, width, channels
        band_predictions = CNN.predict(band_RGB, verbose=0)                                 #Use the CNN model to predict the class of the band
        band_class = np.argmax(band_predictions)                                            #Classify as the class with the highest probability
        extracted_band_classes.append(band_class)                                           #Append the obtained band classification to the return array
    
    return extracted_band_classes                                                       #Return the obtained band classifications



#################################################################################################################################################################################################
### Decode Resistance ###########################################################################################################################################################################     
# ╰┈➤ Decode the resistance and tolerance value of the resistor using the extracted band classifications:

def Decode_Resistance(band_classes, largest_separation_index, tolerance_lookup):
    
    decoded_resistance_num = []                                                         #Init decoded resistance number return array
    decoded_resistance_string = []                                                      #Init decoded resistance string return array
    decoded_tolerance = []                                                              #Init decoded tolerance return array
               
    band_classes = np.array(band_classes)                                               #Convert band classes to numpy array
    num_bands = len(band_classes)                                                       #Find the number of bands
    
    bInvertReadDirection = False                                                        #Assume left to right reading direction
    bAmbiguousReadDirection = True                                                      #Assume reading direction is ambiguous   

    ### 5-Band Decoding --------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Handle the decoding of 5-band resistors. Consider reading direction inversion and ambiguity:
    
    if (num_bands == 5):                                                                #If 5 bands are present:                                    
   
        if ((np.any(band_classes[3:5] > 9))                                                 #If either of the last two bands are metallic (10-11)...
            or (tolerance_lookup[band_classes[0]] == "ERR%")):                              #...or the first band is invalid for a tolerance band:
            bAmbiguousReadDirection = False                                                     #Reading direction is no longer ambiguous       (only the conditions for non-inverted are true)
            
        if ((np.any(band_classes[0:2] > 9))                                                 #If either of the first two bands are metallic (10-11)...
            or (tolerance_lookup[band_classes[4]] == "ERR%")):                              #...or the last band is invalid for a tolerance band:
            
            if (bAmbiguousReadDirection == True):                                               #If reading direction is ambiguous:
                bInvertReadDirection = True                                                         #Invert the reading direction           
                bAmbiguousReadDirection = False                                                     #Reading direction is no longer ambiguous   (only the conditions for inverted are true)
            
            else:                                                                               #Otherwise if the reading direction is not ambiguous:
                bAmbiguousReadDirection = True                                                      #Reading direction is now ambiguous         (the conditions for both directions are true))

        
        ### Forward Reading Direction ------------------------------------------------------#----------------------------#
        if (bInvertReadDirection == False):                                                 #If reading direction is left to right:
            num = int(str(band_classes[0]) + str(band_classes[1]) + str(band_classes[2]))       #Concatenate the first three bands
            mult = band_classes[3]                                                              #Extract the multiplier band
            tolerance = tolerance_lookup[band_classes[4]]                                       #Extract the tolerance band
            
            if (mult > 9): mult = 0 - (mult-9)                                                  #If multiplier is metallic, convert to negative exponent
            
            resistance_num = num * (10.0 ** mult)                                               #Calculate the resistance value
            resistance_string = func.int_to_metric_string(resistance_num)                       #Convert the resistance value to a metric string
            
            decoded_resistance_num.append(resistance_num)                                       #Append results to return array
            decoded_resistance_string.append(resistance_string)
            decoded_tolerance.append(tolerance)

        ### Inverted Reading Direction -----------------------------------------------------#----------------------------#    
        if (bInvertReadDirection == True) or (bAmbiguousReadDirection == True):             #If reading direction is right to left, or ambiguous (need to do both in this case):
            num = int(str(band_classes[4]) + str(band_classes[3]) + str(band_classes[2]))       #Concatenate the last three bands
            mult = band_classes[1]                                                              #Extract the multiplier band
            tolerance = tolerance_lookup[band_classes[0]]                                       #Extract the tolerance band
            
            if (mult > 9): mult = 0 - (mult-9)                                                  #If multiplier is metallic, convert to negative exponent
            
            resistance_num = num * (10.0 ** mult)                                               #Calculate the resistance value     
            resistance_string = func.int_to_metric_string(resistance_num)                       #Convert the resistance value to a metric string
            
            decoded_resistance_num.append(resistance_num)                                       #Append results to return array
            decoded_resistance_string.append(resistance_string)
            decoded_tolerance.append(tolerance)
    
    ### 4-Band Decoding --------------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Handle the decoding of 4-band resistors here. [Not currently implemented]
    elif (num_bands == 4):                                                              #If 4 bands are present:                                
        return (None, None, None, None)                                                     #IMPLEMENT 4-Band decoding here (not required for this project)
                                                                                                                          # (no 4-band resistors to test with)
    ### Reject Other Band Quantities -------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Handle the decoding of 4-band resistors here. [Not currently implemented]
    else:                                                                               #If anything other than 4 or 5 bands are present:                                
        return (None, None, None, None)  
    
    ### Final Ambiguity Tests --------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Check for any remaining ambiguity in the reading direction and resolve if possible:
    
    if (bAmbiguousReadDirection == True):                                               #Attempt to resolve ambiguity by checking if either deocded resistance is 0:
        if (decoded_resistance_num[0] == 0):
            bAmbiguousReadDirection = False
            decoded_resistance_num.pop(0)
            decoded_resistance_string.pop(0)
            decoded_tolerance.pop(0)
        elif (decoded_resistance_num[1] == 0):
            bAmbiguousReadDirection = False
            decoded_resistance_num.pop(1)
            decoded_resistance_string.pop(1)
            decoded_tolerance.pop(1)
    
    if (bAmbiguousReadDirection == True):                                               #Attempt to resolve ambiguity by handling the case of symmetric resistor bands:
        if (decoded_resistance_num[0] == decoded_resistance_num[1]) and (decoded_tolerance[0] == decoded_tolerance[1]):
            bAmbiguousReadDirection = False                 
            decoded_resistance_num.pop(1)
            decoded_resistance_string.pop(1)
            decoded_tolerance.pop(1)

    if (bAmbiguousReadDirection == True):                                               #Attempt to resolve ambiguity by checking E24 and E96 series
        valid_E24 = [func.belongs_to_E24(decoded_resistance_num[0]), func.belongs_to_E24(decoded_resistance_num[1])]
            
        if (valid_E24[0] == True) and (valid_E24[1] == False):
            bAmbiguousReadDirection = False
        elif (valid_E24[0] == False) and (valid_E24[1] == True):
            bAmbiguousReadDirection = False
            bInvertReadDirection = True
                
        elif (valid_E24[0] == False) and (valid_E24[1] == False):
            valid_E96 = [func.belongs_to_E96(decoded_resistance_num[0]), func.belongs_to_E96(decoded_resistance_num[1])]
            
            if (valid_E96[0] == True) and (valid_E96[1] == False):
                bAmbiguousReadDirection = False
            elif (valid_E96[0] == False) and (valid_E96[1] == True):
                bAmbiguousReadDirection = False
                bInvertReadDirection = True
    
    if (bAmbiguousReadDirection == True):                                               #Attempt to resolve ambiguity by checking if either end band is brown (more likely to be tolerance band)
        if (band_classes[-1] == 1) and (band_classes[0] != 1):
            bAmbiguousReadDirection = False
        elif (band_classes[0] == 1) and (band_classes[-1] != 1):
            bAmbiguousReadDirection = False
            bInvertReadDirection = True
 
    #if (bAmbiguousReadDirection == True):                                              #Attempt to resolve ambiguity by checking largest band seperation location (NOT EFFECTIVE)
    #    if (largest_separation_index == num_bands - 2):
    #        bAmbiguousReadDirection = False
    #    elif (largest_separation_index == 0):
    #        bAmbiguousReadDirection = False
    #        bInvertReadDirection = True
    
    if (bAmbiguousReadDirection == True):                                               #If still ambiguous, choose lower value as display one
        if (decoded_resistance_num[0] > decoded_resistance_num[1]): bInvertReadDirection = True
    
    return (decoded_resistance_string, decoded_tolerance, bInvertReadDirection, bAmbiguousReadDirection)


#################################################################################################################################################################################################
### Display Resistance ##########################################################################################################################################################################     
# ╰┈➤ Outline the resistor with its bounding box, and display the decoded resistance and tolerance values to the provided image:

def Display_Resistance(display_image, display_string, bounding_rect):
    
    box = func.Display_Bounding(display_image, bounding_rect, (0,0,255), 10)            #Draw bounding box on image                  

    sorted_box = sorted(box, key=lambda point: point[1])                                # Sort the box points by their y-coordinate in ascending order

    display_pointA = tuple(sorted_box[0])                                               # The two upper-most points are the first two points in the sorted list
    display_pointB = tuple(sorted_box[1])
    
    textsize = cv2.getTextSize(display_string, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.5, thickness=8)[0]

    display_x = int((display_pointA[0] + display_pointB[0])/2) - int(textsize[0]/2)     #Centre text on desired position
    display_y = display_pointA[1] - 20
    display_point = (display_x, display_y)
                                                                                        #Draw text on image
    cv2.putText(display_image, display_string, display_point, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.5, color=(0, 0, 200), thickness=8, lineType=cv2.LINE_AA)  
    
    return display_image



def Export_Bands_For_Training(extracted_bands, target_shape, target_dest, img_index, object_index):
    
    ################################################
    # Export the bands for training
    for i in range(len(extracted_bands)):
        band = extracted_bands[i]
        band_resize = cv2.resize(band, target_shape)
        cv2.imwrite('data_training/' + target_dest + '/' + str(img_index) + '_' + str(object_index) + '_' + str(i) + '.jpg', band_resize)
        print("Exporting band: " + 'data_training/' + target_dest + '/' + str(img_index) + '_' + str(object_index) + '_' + str(i) + '.jpg')
    ################################################)
   
