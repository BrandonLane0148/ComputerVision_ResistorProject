
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



### Median Filter ----------------------------------------------------------------------------------------------
# ╰┈➤ To remove noise but preserve edges
filtered_input_image = cv2.bilateralFilter(input_image,25,25,25)



segmentation_mask                       = sub.Resistor_Segmentation(filtered_input_image)

(extracted_masks, extracted_resistors)  = sub.Extract_Resistors(segmentation_mask, filtered_input_image, min_resistor_res, target_resistor_width)

for i in range(len(extracted_resistors)):
    mask = extracted_masks[i]
    resistor = extracted_resistors[i]
    
    refined_mask = sub.Refine_Resistor_Mask(mask)
    resistor = cv2.bitwise_and(resistor, resistor, mask = refined_mask)
    
    inner_resistor = sub.Inner_Resistor_Extraction(resistor, refined_mask, target_resistor_width)
    if (inner_resistor is None): continue #HANDLE THIS BETTER

    (band_mask, bodycolour_abs_diff) = sub.Band_Segmentation(inner_resistor)
    inner_resistor_bands = cv2.bitwise_and(inner_resistor, inner_resistor, mask = band_mask)
    



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