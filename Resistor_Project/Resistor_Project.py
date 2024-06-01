
from dataclasses import MISSING
import cv2 #opencv itself
import numpy as np # matrix manipulations
from tensorflow.keras.models import load_model

import Functions as func
import Subroutines as sub

###########################################################
### Initialisation
colour_lookup =    ["Black", "Brown", "Red" , "Orange", "Yellow", "Green", "Blue" , "Violet", "Grey" , "White", "Gold", "Silver"]
tolerance_lookup = ["1%"   , "2%"   , "ERR%", "ERR%"  , "ERR%"  , "0.5%" , "0.25%", "0.1%"  , "0.05%", "ERR%" , "5%"  , "10%"   ]

num_input_images = 30
input_resistor_values = [["220"], ["5k1"], ["10k"], ["10"], ["1M"], ["330"], ["2k"], ["5k1"], ["330"], ["220"],\
                         ["220", "10"], ["2k", "330"], ["5k1", "10k"], ["330", "220"], ["100", "1M"], ["220"], ["330"], ["2k"], ["5k1"], ["10k"]]
#max_input_res = [2000, 3000]
max_input_res = [1440, 2560/2]
#max_input_res = [1080, 1920/2]

min_resistor_res = 20
target_resistor_width = 300
target_band_shape = (12, 30)

#knn = cv2.ml.KNearest_create() #Create the K-NN model
#sub.Train_KNN_Classifier(knn) #Train the K-NN model

# Load the trained CNN model
Colour_Classification_CNN = load_model('models/Colour_Classification_CNN.keras')

###########################################################
### Image Input

#indexes to fix/consider: 95, 96, 97, 98, 103    
    
input_index = 31
input_image = cv2.imread('data_training_extraction/testing/'+str(input_index)+'.jpg')
#input_image = cv2.imread('data/ideal.jpg')
#input_image = cv2.imread('data/import.jpg')

#Resize if required:
if (input_image.shape[0] > max_input_res[0] or input_image.shape[1] > max_input_res[1]):
    h_scale = input_image.shape[0] / max_input_res[0]
    w_scale = input_image.shape[1] / max_input_res[1]
    
    scale = max(h_scale, w_scale)
    new_dim = [int(input_image.shape[1] / scale), int(input_image.shape[0] / scale)]
    input_image = cv2.resize(input_image, new_dim)

cv2.imshow("Input", input_image)
cv2.waitKey(0) # waits for user to press any key 

###########################################################
### Blur
image_blur = cv2.bilateralFilter(input_image,25,25,25)

cv2.imshow("Test", image_blur)
cv2.waitKey(0) # waits for user to press any key

################################################
### Gamma Correct
image_gamma_correct = func.applyGammaCorrection(image_blur, 3) #Consider opposite gamma correction for black background

cv2.imshow("Test", image_gamma_correct)
cv2.waitKey(0) # waits for user to press any key 

################################################
### OTSU Threshold on HSV 'S' Channel
image_HSV = cv2.cvtColor(image_gamma_correct, cv2.COLOR_BGR2HSV)
otsu_threshold, mask_OTSU = cv2.threshold(image_HSV[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
#scale and manually use otsu threshold for more aggressive segmentation?

cv2.imshow("Test", image_HSV[:, :, 1])
cv2.waitKey(0) # waits for user to press any key

cv2.imshow("Test", mask_OTSU)
cv2.waitKey(0) # waits for user to press any key
################################################
### Erode Mask to remove wires/noise
erosion_dist = 1
erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_dist + 1, 2 * erosion_dist + 1))
mask_eroded = cv2.erode(mask_OTSU, erosion_kernel, iterations = 2)

cv2.imshow("Test", mask_eroded)
cv2.waitKey(0) # waits for user to press any key 
################################################
### Dilate Mask to shrink gaps
dilation_dist = 1
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_dist + 1, 2 * dilation_dist + 1))
mask_dilated = cv2.dilate(mask_eroded, dilation_kernel, iterations = 2)

cv2.imshow("Test", mask_dilated)
cv2.waitKey(0) # waits for user to press any key 
################################################
### Blur mask to merge outlines
mask_blur = cv2.GaussianBlur(mask_dilated, (21,21), 5)
cv2.imshow("Test", mask_blur)
cv2.waitKey(0) # waits for user to press any key 
################################################
### Find contours of mask
contours, hierarchy = cv2.findContours(mask_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask_contours = cv2.cvtColor(mask_eroded.copy(), cv2.COLOR_GRAY2BGR)
cv2.drawContours(mask_contours, contours, -1, (0,255,0), 2)

cv2.imshow("Test", mask_contours)
cv2.waitKey(0) # waits for user to press any key 
################################################
### Find resistor bounding boxes using contours
mask_boundings = mask_contours.copy()
bounding_rects = []
for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    dim = rect[1]   #Enforce min res to remove stray noise
    if (min(dim) < min_resistor_res): continue
    
    cv2.drawContours(mask_boundings,[box],0,(0,0,255),3)
    bounding_rects.append(rect)

cv2.imshow("Test", mask_boundings)
cv2.waitKey(0) # waits for user to press any key 
################################################
### Extract resistors using bounding boxes
extracted_masks = []
extracted_resistors = []
for rect in bounding_rects:
    mask = func.extractImgUsingRect(mask_eroded.copy(), rect, target_width = target_resistor_width)
    resistor = func.extractImgUsingRect(image_blur.copy(), rect, target_width = target_resistor_width)
    
    extracted_masks.append(mask)
    extracted_resistors.append(resistor)

for i in range(len(extracted_masks)):
    cv2.imshow(str(i), extracted_masks[i])
cv2.waitKey(0) # waits for user to press any key 
################################################
### Fix masks using morphological operations:
extracted_masks_final = sub.ResolveMasks_method1(extracted_masks)

for i in range(len(extracted_resistors)):
    extracted_resistors[i] = cv2.bitwise_and(extracted_resistors[i], extracted_resistors[i], mask = extracted_masks_final[i])
    cv2.imshow(str(i), extracted_resistors[i])
cv2.waitKey(0) # waits for user to press any key 
################################################
### Extract inner cutout of resistor body/bands
extracted_resistors_inner = []
for i in range(len(extracted_resistors)):
    resistor = extracted_resistors[i].copy()
    mask = extracted_masks_final[i].copy()
   
    h, w = mask.shape
    mid_h = int(h/2)
    mid_w = int(w/2)
    rng = int(target_resistor_width/10)

    y1, y2 = -1, -1
    for i in range(mid_h-1):
        if (np.all(mask[i, mid_w-rng:mid_w+rng] == 255)) and (y1 == -1):
            y1 = i
        if (np.all(mask[h-i-1, mid_w-rng:mid_w+rng] == 255)) and (y2 == -1):
            y2 = h-i-1
        if (y1 != -1 and y2 != -1):
            continue
    if (y1 == -1 or y2 == -1): continue
    
    h_reduce = 0.4
    h_dist = y2 - y1
    y1 = int(y1 + ((h_dist * h_reduce)/2))
    y2 = int(y2 - ((h_dist * h_reduce)/2))
    
    x1, x2 = -1, -1
    for i in range(mid_w-1):
        if (np.all(mask[y1:y2, i] == 255)) and (x1 == -1):
            x1 = i
        if (np.all(mask[y1:y2, w-i-1] == 255)) and (x2 == -1):
            x2 = w-i-1
        if (x1 != -1 and x2 != -1):
            continue
    if (x1 == -1 or x2 == -1): continue
    
    w_reduce = 0.0
    w_dist = x2 - x1
    x1 = int(x1 + ((w_dist * w_reduce)/2))
    x2 = int(x2 - ((w_dist * w_reduce)/2))
    
    if (h_dist >= 16):
        inner_resistor = resistor[y1:y2, x1:x2]  
        extracted_resistors_inner.append(inner_resistor)
    
for i in range(len(extracted_resistors_inner)):
    cv2.imshow(str(i) + "a", extracted_resistors_inner[i])
cv2.waitKey(0) # waits for user to press any key 
################################################
### Extract body colour as most common K-Means colour
extracted_resistors_bodycolour = []
for resistor_inner in extracted_resistors_inner:   
    h, w, _ = resistor_inner.shape
    mid_h = int(h/2)
    img_bodycolour = resistor_inner.copy()
    img_bodycolour[:] = func.ExtractDominantColour(resistor_inner[mid_h-7:mid_h+7, :], 5, 60)
    extracted_resistors_bodycolour.append(img_bodycolour)
    
for i in range(len(extracted_resistors_bodycolour)):
    cv2.imshow(str(i) + "b", extracted_resistors_bodycolour[i])
cv2.waitKey(0) # waits for user to press any key 
################################################
### Subtract body colour from resistor
extracted_resistors_inner_difference = []
for i in range(len(extracted_resistors_inner)):
    abs_diff = cv2.absdiff(extracted_resistors_inner[i], extracted_resistors_bodycolour[i])
    extracted_resistors_inner_difference.append(abs_diff)
    cv2.imshow(str(i) + "b", abs_diff)
cv2.waitKey(0) # waits for user to press any key 
################################################
### OTSU threshold the absdiff using HSV 'V' Channel
extracted_resistors_inner_bandsmask = []
for diff in extracted_resistors_inner_difference:
    diff_HSV = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
    otsu_threshold, mask_OTSU = cv2.threshold(diff_HSV[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(diff_HSV[:, :, 2], int(otsu_threshold * 0.9), 255, cv2.THRESH_BINARY)
       
    extracted_resistors_inner_bandsmask.append(mask)

for i in range(len(extracted_resistors_inner_bandsmask)):
    cv2.imshow(str(i) + "b", extracted_resistors_inner_bandsmask[i])
cv2.waitKey(0) # waits for user to press any key 
################################################
### Dilate and erode resulting band mask
for i in range(len(extracted_resistors_inner_bandsmask)):
    dilation_dist = 2
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_dist + 1, 2 * dilation_dist + 1))
    extracted_resistors_inner_bandsmask[i] = cv2.dilate(extracted_resistors_inner_bandsmask[i], dilation_kernel, iterations = 1)
    #test dilating with column kernel instead

    erosion_dist = 1
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_dist + 1, 2 * erosion_dist + 1))
    extracted_resistors_inner_bandsmask[i] = cv2.erode(extracted_resistors_inner_bandsmask[i], erosion_kernel, iterations = 3)

    cv2.imshow(str(i) + "b", extracted_resistors_inner_bandsmask[i])
cv2.waitKey(0) # waits for user to press any key
################################################
### Only keep purely white vertical slices
extracted_resistors_inner_bandsmask_final = []
for mask in extracted_resistors_inner_bandsmask:
    h, w = mask.shape
    leeway = 5
    mask_final = mask.copy()
    for i in range(w):
        if (np.any(mask[leeway:h-leeway, i] == 0)):
            mask_final[0:h, i] = 0
        else:
            mask_final[0:h, i] = 255
    extracted_resistors_inner_bandsmask_final.append(mask_final)

for i in range(len(extracted_resistors_inner_bandsmask_final)):
    cv2.imshow(str(i) + "b", extracted_resistors_inner_bandsmask_final[i])
cv2.waitKey(0) # waits for user to press any key 
################################################
### Apply band mask
extracted_resistors_bands_masked = []
for i in range(len(extracted_resistors_inner)):
    resistor_bands = cv2.bitwise_and(extracted_resistors_inner[i], extracted_resistors_inner[i], mask = extracted_resistors_inner_bandsmask_final[i])
    extracted_resistors_bands_masked.append(resistor_bands)
    cv2.imshow(str(i) + "b", resistor_bands)
cv2.waitKey(0) # waits for user to press any key 
################################################
### Obtain band info (start, end, width) from mask
extracted_bands_info = []
for i in range(len(extracted_resistors_inner_bandsmask_final)):
    mask = extracted_resistors_inner_bandsmask_final[i].copy()
    h, w = mask.shape
    mid_h = int(h/2)
    
    min_width = 3
    
    bands_info = []
    bBand = False
    x0 = 0
    for j in range(w):
        if (mask[mid_h, j] == 255) and (bBand == False):
            bBand = True
            x0 = j
        if (mask[mid_h, j] == 0) and (bBand == True):
            bBand = False
            band_info = [x0, j, j-x0]  ##Band start, band end, band width
            if band_info[2] > min_width:
                bands_info.append(band_info)
        if (j == w-1) and (bBand == True):
            band_info = [x0, w, w-x0]
            if band_info[2] > min_width:
                bands_info.append(band_info)
    if (len(bands_info) >= 3):
        ##############################################################
        ### Extrapolate Missing Bands
        
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
        extracted_bands_info.append(bands_info)
    else: extracted_resistors_inner.pop(i)
        
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
################################################
### Export bands for CNN Training

#print("\n\n")
#sub.Export_Bands_For_Training(extracted_bands, target_band_shape, "training", input_index)        
#cv2.waitKey(0)

################################################
### Classify Band Colours using the CNN model
     
extracted_bands_classes = sub.Classify_Band_Colours_CNN(extracted_bands, Colour_Classification_CNN)


################################################
### Decode Resistance
sub.DecodeResistance(extracted_bands_classes, tolerance_lookup)
cv2.waitKey(0)


# closing all open windows 
cv2.destroyAllWindows() 