#DILATION/EROSION

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

extracted_masks_final = []
for mask in extracted_masks_dilated:
    eroded = mask.copy()
    
    eroded = cv2.erode(eroded, erosion_kernel, iterations = 2)
    eroded = cv2.erode(eroded, erosion_kernel_flat, iterations = 10)
    #eroded = cv2.erode(eroded, erosion_kernel, iterations = 4)
    extracted_masks_final.append(eroded)
    
for i in range(len(extracted_masks_final)):
    cv2.imshow(str(i), extracted_masks_final[i])
cv2.waitKey(0) # waits for user to press any key 
################################################



#CLOSING


################################################

closing_dist = 3
closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * closing_dist + 1, 2 * closing_dist + 1))

extracted_masks_closed = []
for mask in extracted_masks_joined:
   
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel, iterations = 2)

    extracted_masks_closed.append(closed)
    
for i in range(len(extracted_masks_closed)):
    cv2.imshow(str(i), extracted_masks_closed[i])
cv2.waitKey(0) # waits for user to press any key 

################################################



#FISHBONE:

################################################

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



#EXTENDED FISHBONE FULL:

################################################
    
for i in range(len(extracted_masks)):
    _, extracted_masks[i] = cv2.threshold(extracted_masks[i],100,255,cv2.THRESH_BINARY)
    cv2.imshow(str(i), extracted_masks[i])
cv2.waitKey(0) # waits for user to press any key 
################################################

closing_dist = 3
closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * closing_dist + 1, 2 * closing_dist + 1))
extracted_masks_closed = []
for mask in extracted_masks:
   
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel, iterations = 1)

    extracted_masks_closed.append(closed)
    
for i in range(len(extracted_masks_closed)):
    cv2.imshow(str(i), extracted_masks_closed[i])
cv2.waitKey(0) # waits for user to press any key 
################################################

extracted_masks_joined = extracted_masks_closed.copy()
for mask in extracted_masks_joined:
    h, w = mask.shape
    mid = int(h/2)
    x1, x2 = -1, -1
    for i in range(w):
        if (mask[mid, i] > 200) and (x1 == -1):
            x1 = -2
        if (mask[mid, i] < 200) and (x1 == -2):
            x1 = i
        if (mask[mid, w-i-1] > 200) and (x2 == -1):
            x2 = -2
        if (mask[mid, w-i-1] < 200) and (x2 == -2):
            x2 = w-i-1
        if (x1 > 0 and x2 > 0):
            continue
    mask[mid-25:mid+25, x1:x2] = 255
        
for i in range(len(extracted_masks_joined)):
    cv2.imshow(str(i), extracted_masks_joined[i])
cv2.waitKey(0) # waits for user to press any key 


################################################

extracted_masks_final = extracted_masks_closed.copy()
for i in range(len(extracted_resistors)):
    extracted_resistors[i] = cv2.bitwise_and(extracted_resistors[i], extracted_resistors[i], mask = extracted_masks_final[i])
    cv2.imshow(str(i), extracted_resistors[i])
cv2.waitKey(0) # waits for user to press any key 
################################################


#HSV SOBEL ANALYSIS:

################################################

extracted_resistors_edges = []
for resistor in extracted_resistors:
    hsv = cv2.cvtColor(resistor, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv[:, :, 0], (11,11), 5)
    sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
    sobelx_8u = cv2.convertScaleAbs(sobelx)
    extracted_resistors_edges.append(sobelx_8u)
    
for i in range(len(extracted_resistors_edges)):
    cv2.imshow("H: " + str(i), extracted_resistors_edges[i])
cv2.waitKey(0) # waits for user to press any key 

extracted_resistors_edges = []
for resistor in extracted_resistors:
    hsv = cv2.cvtColor(resistor, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv[:, :, 1], (11,11), 5)
    sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
    sobelx_8u = cv2.convertScaleAbs(sobelx)
    extracted_resistors_edges.append(sobelx_8u)
    
for i in range(len(extracted_resistors_edges)):
    cv2.imshow("S: " + str(i), extracted_resistors_edges[i])
cv2.waitKey(0) # waits for user to press any key 

extracted_resistors_edges = []
for resistor in extracted_resistors:
    hsv = cv2.cvtColor(resistor, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv[:, :, 2], (11,11), 5)
    sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
    sobelx_8u = cv2.convertScaleAbs(sobelx)
    extracted_resistors_edges.append(sobelx_8u)
    
for i in range(len(extracted_resistors_edges)):
    cv2.imshow("V: " + str(i), extracted_resistors_edges[i])
cv2.waitKey(0) # waits for user to press any key 

extracted_resistors_edges = []
for resistor in extracted_resistors:
    hsv = cv2.cvtColor(resistor, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (11,11), 5)
    sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
    sobelx_8u = cv2.convertScaleAbs(sobelx)
    extracted_resistors_edges.append(sobelx_8u)
    
for i in range(len(extracted_resistors_edges)):
    cv2.imshow("HSV: " + str(i), extracted_resistors_edges[i])
cv2.waitKey(0) # waits for user to press any key 
#################################################################


#K-MEANS WITH PREDEFINED COLOUR PALETTE

centers = np.array([[B1, G1, R1], [B2, G2, R2], ..., [Bk, Gk, Rk]], dtype=np.float32)
pixels = image.reshape(-1, 3)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, _ = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_USE_INITIAL_LABELS, centers)
quantized_image = centers[labels.flatten()]
quantized_image = np.reshape(quantized_image, image.shape)


######################################################

#Eucliden distance clustering using CIELAB

# Define your color palette
palette = np.array([[L1, a1, b1], [L2, a2, b2], ..., [Lk, ak, bk]], dtype=np.float32)

# Load your image
image = cv2.imread('image.jpg')

# Convert the image to the CIELAB color space
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# Reshape the image to a 2D array
pixels = image_lab.reshape(-1, 3)

# Calculate the Euclidean distance between each pixel and the color palette
distances = np.linalg.norm(pixels[:, None] - palette, axis=2)

# Assign each pixel to the closest color in the palette
labels = np.argmin(distances, axis=1)

# Replace each pixel with its label color
classified_image = palette[labels].reshape(image.shape).astype(np.uint8)

# Convert the classified image back to the BGR color space for display
classified_image_bgr = cv2.cvtColor(classified_image, cv2.COLOR_Lab2BGR)



###################################################

#KNN training/init

# Load your training image
training_image = cv2.imread('training_image.jpg')

# Convert the training image to the CIELAB color space
training_image_lab = cv2.cvtColor(training_image, cv2.COLOR_BGR2Lab)

# Reshape the training image to a 2D array
training_data = training_image_lab.reshape(-1, 3)

# Create the labels for the training data
labels = np.repeat(np.arange(training_image_lab.shape[0]), training_image_lab.shape[1])

# Create the K-NN model
knn = cv2.ml.KNearest_create()

# Train the K-NN model
knn.train(training_data, cv2.ml.ROW_SAMPLE, labels)


###################################################

#KNN Classification

# Define your color value in the BGR color space
color_bgr = np.array([[[B, G, R]]], dtype=np.uint8)

# Convert the color value to the CIELAB color space
color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2Lab)

# Reshape the color value to a 2D array
color_data = color_lab.reshape(-1, 3)

# Find the nearest neighbor
_, results, _, _ = knn.findNearest(color_data, k=1)

# The result is the label of the nearest neighbor
label = results[0, 0]

###################################################

#Deprecated body colour extraction method

def ExtractBodyColour_method1(resistor_inner):
    h, w, _ = resistor_inner.shape

    bodycolour_samples = (0, 0, 0)
    bodycolour_samples += (resistor_inner[0,   0])
    bodycolour_samples += (resistor_inner[h-1, 0])
    bodycolour_samples += (resistor_inner[0,   w-1])
    bodycolour_samples += (resistor_inner[h-1, w-1])
    
    bodycolour = (int(bodycolour_samples[0] / 4), int(bodycolour_samples[1] / 4), int(bodycolour_samples[2] / 4))
    
    return bodycolour



### Enhance Colours
#for i in range(len(extracted_resistors_bands)):
    #extracted_resistors_bands[i] = func.ImageColourEnhance_CLAHE(extracted_resistors_bands[i], (5,5))
    #cv2.imshow(str(i) + "b", extracted_resistors_bands[i])
#cv2.waitKey(0) # waits for user to press any key 
################################################



#BACKUP BEFORE TESTING MISSING BAND INSERTION

################################################
### Only keep purely white vertical slices
extracted_resistors_inner_bandsmask_final = []
for mask in extracted_resistors_inner_bandsmask:
    h, w = mask.shape
    leeway = 7
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
extracted_resistors_bands = []
for i in range(len(extracted_resistors_inner)):
    resistor_bands = cv2.bitwise_and(extracted_resistors_inner[i], extracted_resistors_inner[i], mask = extracted_resistors_inner_bandsmask_final[i])
    extracted_resistors_bands.append(resistor_bands)
    cv2.imshow(str(i) + "b", resistor_bands)
cv2.waitKey(0) # waits for user to press any key 
################################################
### Extract bands
extracted_bands = []
for i in range(len(extracted_resistors_bands)):
    resistor_bands = extracted_resistors_bands[i].copy()
    mask = extracted_resistors_inner_bandsmask_final[i].copy()
    h, w = mask.shape
    mid_h = int(h/2)
    
    bands = []
    bBand = False
    x0 = 0
    for j in range(w):
        if (mask[mid_h, j] == 255) and (bBand == False):
            bBand = True
            x0 = j
        if (mask[mid_h, j] == 0) and (bBand == True):
            bBand = False
            band = resistor_bands[0:h, x0:j-1]
            if band.size != 0:
                bands.append(band)
        if (j == w-1) and (bBand == True):
            band = resistor_bands[0:h, x0:j]
            if band.size != 0:
                bands.append(band)
    if (len(bands) > 0):
        extracted_bands.append(bands)
cv2.waitKey(0)
################################################
### Export bands for CNN Training





################################################
### DEPRECATED K-NN SUBROUTINES

def Train_KNN_Classifier(knn):
    # Load your training image
    training_image = cv2.imread('data/training_data.jpg')

    # Convert the training image to the CIELAB color space
    training_image_CIELAB = cv2.cvtColor(training_image, cv2.COLOR_BGR2Lab)

    # Reshape the training image to a 2D array
    training_data = training_image_CIELAB.reshape(-1, 3).astype(np.float32)
    #training_data = training_image_CIELAB.reshape(-1, 3)[:, 1:].astype(np.float32)

    # Create the labels for the training data
    labels = np.repeat(np.arange(training_image_CIELAB.shape[0]), training_image_CIELAB.shape[1])

    # Train the K-NN model
    knn.train(training_data, cv2.ml.ROW_SAMPLE, labels)
    
def Classify_Band_Colours_KNN(extracted_bands, extracted_resistors, knn):
    ################################################
    ### Extract band colour
    bDisplayForTrainingDataAquisition = True
            
    extracted_bands_colours = []
    for bands in extracted_bands:
        colours = []
        if (len(bands) == 0): continue
        # Apply weak Gaussian blur to the band
        for band in bands:
            #band_blur = cv2.GaussianBlur(band, (5, 5), 3)
            colour = func.ExtractDominantColour(band, 1)
            colours.append(colour)
            if (bDisplayForTrainingDataAquisition == True): print(colour)
        extracted_bands_colours.append(colours)

    if (bDisplayForTrainingDataAquisition == True):
        for i in range(len(extracted_bands_colours)):
            preview = extracted_resistors[i].copy()
            preview[:] = (0, 0, 0)
            for j in range(len(extracted_bands_colours[i])):
                colour = extracted_bands_colours[i][j]           
                preview[:, j*60:((j+1)*60)-10] = colour
            cv2.imshow(str(i), preview)
        cv2.waitKey(0) # waits for user to press any key 
    ################################################
    ### Classify band colours
    extracted_bands_classes = []
    for bands in extracted_bands_colours:
        band_classes = []
        for band_colour in bands:
            band_colour = band_colour.reshape(1, 1, 3).astype(np.uint8)
            # Convert the color value to the CIELAB color space
            band_color_CIELAB = cv2.cvtColor(band_colour, cv2.COLOR_BGR2Lab)

            # Reshape the color value to a 2D array
            color_data = band_color_CIELAB.reshape(-1, 3).astype(np.float32)
            #color_data = band_color_CIELAB.reshape(-1, 3)[:, 1:].astype(np.float32)

            # Find the nearest neighbor
            _, results, _, _ = knn.findNearest(color_data, k=1)

            # The result is the label of the nearest neighbor
            label = results[0, 0]
            band_classes.append(int(label))
        extracted_bands_classes.append(band_classes)
    print(extracted_bands_classes)
    cv2.waitKey(0) # waits for user to press any key  
    
    return extracted_bands_classes




##############################################################
        ### Extrapolate Missing Bands
        if (len(bands_info) < 5) and (len(bands_info) > 2):
            band_seperations = []
            band_widths = []
            for i in range(len(bands_info) - 1):
                band_seperations.append(bands_info[i+1][0] - bands_info[i][1])
                band_widths.append(bands_info[i][2])
            band_widths.append(bands_info[-1][2])
        
            #Find the largest seperation between bands
            largest_separation_index = np.argmax(band_seperations)
            largest_separation = band_seperations[largest_separation_index]
        
            #Find the average band seperation EXCLUDING the largest seperation
            typical_seperations = band_seperations.copy()
            typical_seperations.pop(largest_separation_index)
            ave_seperation = np.mean(typical_seperations)
        
            #If the largest seperation is more than twice the average seperation, then insert a band inbetween the two
            if (largest_separation >= 1.79 * ave_seperation):
                #Find the middle point between the two bands
                x0 = bands_info[largest_separation_index][1] + int(largest_separation/2)
                #Find the average band width
                ave_band_width = np.mean(band_widths)
                #Obtain target band width as 0.75 * average band width
                target_band_width = int(0.75 * ave_band_width)
                #Insert the new band with its centre at x0
                bands_info.insert(largest_separation_index+1, [x0 - int(target_band_width/2), x0 + int(target_band_width/2), target_band_width])
                print("Inserted band at: ", x0)
                    
            start_seperation = bands_info[0][0]     #Find the gap between the image start and the first band
            end_seperation = w - bands_info[-1][1]  #Find the gap between the image end and the last band
        
            #If the gap between the image start and first band is >1.75x the average seperation, then insert a band:
            if (start_seperation >= 1.8 * ave_seperation):
                #Find the middle point between the image start and the first band
                x0 = 0 + int(start_seperation/2)
                #Find the average band width
                ave_band_width = np.mean(band_widths)
                #Obtain target band width as 0.75 * average band width
                target_band_width = int(0.75 * ave_band_width)
                #Insert the new band with its centre at x0
                bands_info.insert(0, [x0 - int(target_band_width/2), x0 + int(target_band_width/2), target_band_width])
                print("Inserted band at start")
            
            #If the gap between the image end and the last band is >1.75x the average seperation, then insert a band:
            if (end_seperation >= 1.8 * ave_seperation):
                #Find the middle point between the image end and the last band
                x0 = bands_info[-1][1] + int(end_seperation/2)
                #Find the average band width
                ave_band_width = np.mean(band_widths)
                #Obtain target band width as 0.75 * average band width
                target_band_width = int(0.75 * ave_band_width)
                #Insert the new band with its centre at x0
                bands_info.append([x0 - int(target_band_width/2), x0 + int(target_band_width/2), target_band_width])
                print("Inserted band at end")   
                


def ImageColourEnhance(img):
    #Convert to YCrCb colour space and seperate channels:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    
    #Equalise histogram of Y channel and combine back together:
    cv2.equalizeHist(channels[0], channels[0]) 
    ycrcb = cv2.merge(channels)
    
    #Revert to RGB and return result:
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img

def ImageColourEnhance_CLAHE(img, size):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=size)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    
    return img2


def LocateMissingBand(xm, rng, target_band_width, abs_diff_img):
    #Obtain the image height
    h = abs_diff_img.shape[0] 
    
    #Overestimate band start and end
    x0 = xm - int(rng/2)
    x1 = xm + int(rng/2)
                
    #Recalulate band mask about predicted band location for both H and V channels
    diff_HSV = cv2.cvtColor(abs_diff_img[:, x0:x1], cv2.COLOR_BGR2HSV)
    otsu_threshold, _ = cv2.threshold(diff_HSV[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_H = cv2.threshold(diff_HSV[:, :, 0], int(otsu_threshold * 0.8), 255, cv2.THRESH_BINARY)
    otsu_threshold, _ = cv2.threshold(diff_HSV[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_V = cv2.threshold(diff_HSV[:, :, 2], int(otsu_threshold * 0.8), 255, cv2.THRESH_BINARY)
    #cv2.imshow("Preview", mask)
                
    #Find mask column with largest average value for both H and V channels
    col_sum_H = np.sum(mask_H, axis = 0)
    x_max_H = np.argmax(col_sum_H)
    col_sum_V = np.sum(mask_V, axis = 0)
    x_max_V = np.argmax(col_sum_V)
                   
    #Find the continuity of the max columns
    continuity_H = col_sum_H[x_max_H]/(h*255)
    continuity_V = col_sum_V[x_max_V]/(h*255)
    
    #Select the appropriate channel: H or V
    if (continuity_H > continuity_V):
        col_sum = col_sum_H
        x_max = x_max_H
        continuity = continuity_H
    else:
        col_sum = col_sum_V
        x_max = x_max_V
        continuity = continuity_V
    
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
