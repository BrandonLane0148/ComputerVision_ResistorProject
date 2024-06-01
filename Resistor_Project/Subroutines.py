import cv2 #opencv itself
import numpy as np # matrix manipulations
    
from matplotlib import pyplot as plt # this lets you draw inline pictures in the notebooks
import pylab # this allows you to control figure size 

import Functions as func

def ResolveMasks_method1(extracted_masks):
    
    extracted_masks_joined = extracted_masks.copy()
    for mask in extracted_masks_joined:
        h, w = mask.shape
        mid = int(h/2)
        x1, x2 = -1, -1
        for i in range(w):
            if (mask[mid, i] > 200) and (x1 == -1):
                x1 = i
            if (mask[mid, w-i-1] > 200) and (x2 == -1):
                x2 = w-i-1
            if (x1 != -1 and x2 != -1):
                continue
        mask[mid-5:mid+5, x1:x2] = 255
        
    for i in range(len(extracted_masks_joined)):
        cv2.imshow(str(i), extracted_masks_joined[i])
    cv2.waitKey(0) # waits for user to press any key 
    ################################################
    
    for i in range(len(extracted_masks_joined)):
        _, extracted_masks_joined[i] = cv2.threshold(extracted_masks_joined[i],100,255,cv2.THRESH_BINARY)
        cv2.imshow(str(i), extracted_masks_joined[i])
    cv2.waitKey(0) # waits for user to press any key 
    ################################################

    dilation_dist = 2
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_dist + 1, 2 * dilation_dist + 1))
    #dilation_kernel = np.ones((1, 2 * dilation_dist + 1))
    dilation_kernel_up = np.array([[1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0]], dtype="uint8")
    dilation_kernel_down = np.roll(dilation_kernel_up, 2, axis=0)

    extracted_masks_dilated = []
    for mask in extracted_masks_joined:
        h, w = mask.shape
        mid = int(h/2)
        dilated = mask.copy()
    
        dilated[0:mid, :] = cv2.dilate(mask[0:mid, :], dilation_kernel_up, iterations = 10)
        dilated[mid:h-1, :] = cv2.dilate(mask[mid:h-1, :], dilation_kernel_down, iterations = 10)
        #dilated = cv2.dilate(mask, dilation_kernel, iterations = 5)
        extracted_masks_dilated.append(dilated)
    
    for i in range(len(extracted_masks_dilated)):
        cv2.imshow(str(i), extracted_masks_dilated[i])
    cv2.waitKey(0) # waits for user to press any key 
    ################################################

    erosion_dist = 2
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_dist + 1, 2 * erosion_dist + 1))
    erosion_kernel_flat = np.ones((1, 2 * dilation_dist + 1))

    extracted_masks_eroded = []
    for mask in extracted_masks_dilated:
        eroded = mask.copy()
    
        eroded = cv2.erode(eroded, erosion_kernel, iterations = 2)
        eroded = cv2.erode(eroded, erosion_kernel_flat, iterations = 10)
        #eroded = cv2.erode(eroded, erosion_kernel, iterations = 4)
        extracted_masks_eroded.append(eroded)
    
    for i in range(len(extracted_masks_eroded)):
        cv2.imshow(str(i), extracted_masks_eroded[i])
    cv2.waitKey(0) # waits for user to press any key 
    ################################################
    
    closing_dist = 3
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * closing_dist + 1, 2 * closing_dist + 1))

    extracted_masks_final = []
    for mask in extracted_masks_eroded:
   
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel, iterations = 2)

        extracted_masks_final.append(closed)
    
    #extracted_masks_final = extracted_masks_eroded.copy()
    
    for i in range(len(extracted_masks_final)):
        cv2.imshow(str(i), extracted_masks_final[i])
    cv2.waitKey(0) # waits for user to press any key 
    
    return extracted_masks_final

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