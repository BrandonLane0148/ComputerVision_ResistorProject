
import cv2 #opencv itself
import numpy as np # matrix manipulations
from tensorflow.keras.models import load_model
import keyboard

import Functions as func
import Subroutines as sub

#################################################################################################################################################################################################
### Initialisation ##############################################################################################################################################################################     
# ╰┈➤ Initialise various parameters, lookup tables, and load the CNN model:

colour_lookup =    ["Black", "Brown", "Red" , "Orange", "Yellow", "Green", "Blue" , "Violet", "Grey" , "White", "Gold", "Silver"]
tolerance_lookup = ["ERR%" , "1%"   , "2%"  , "ERR%"  , "ERR%"  , "0.5%" , "0.25%", "0.1%"  , "0.05%", "ERR%" , "5%"  , "10%"   ]

num_input_images = 44
input_resistor_values = [["220"], ["5k1"], ["10k"], ["10"], ["1M"], ["330"], ["2k"], ["5k1"], ["330"], ["220"],\
                         ["220", "10"], ["2k", "330"], ["5k1", "10k"], ["330", "220"], ["100", "1M"], ["220"], ["330"], ["2k"], ["5k1"], ["10k"]]

max_input_res = [2560, 2560]
max_display_res = [1440, 2560]


min_resistor_res = 20
target_resistor_width = 300
target_band_shape = (12, 30)

bExportBandsForTraining = False

# Load the trained CNN model
Colour_Classification_CNN = load_model('models/Colour_Classification_CNN.keras')

#################################################################################################################################################################################################
### Main Program Loop ###########################################################################################################################################################################     
# ╰┈➤ Loop through all input images, and perform the resistor decoding process:

input_index = 1

while (True):
    
    ### Load Input Image -------------------------------------------------------------------------------------------------------------------------------------------------
    input_image = cv2.imread('data_training_extraction/testing/'+str(input_index)+'.jpg')

    ### Preprocess Input Image -------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Preprocess the input image to resize and filter out noise:
    preprocessed_input_image                = sub.Input_Image_Preprocessing(input_image, max_input_res)

    func.displayImage("Resistor Decoder", preprocessed_input_image, max_display_res)
    
    ### Wait for User Input -----------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Start processing image after user input, and show a "Processing..." message:
    
    status_message = ""
    bSkip = False
    bQuit = False
    while (True):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('.'):  # Right arrow key
            input_index += 1
            if (input_index > num_input_images): input_index = 0
            bSkip = True
            status_message = "Skipping to next image..."
            break
        elif key == ord(','):  # Left arrow key
            input_index -= 1
            if (input_index < 0): input_index = num_input_images
            bSkip = True
            status_message = "Skipping to previous image..."
            break
        elif key == 27:  # Esc key
            bQuit = True
            status_message = "Quitting..."
            break
        elif key != 255:   # Any other key
            status_message = "Processing..."
            break
        
    status = preprocessed_input_image.copy()
    cv2.putText(status, status_message, (35, 75), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.5, color=(0, 0, 255), thickness=8, lineType=cv2.LINE_AA)
    func.displayImage("Resistor Decoder", status, max_display_res)
    cv2.waitKey(1)  # Force window update
    
    if (bSkip): continue
    if (bQuit): break

    ### Resistor Segmentation --------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Isolate the resistors from the background: 
    segmentation_mask                       = sub.Resistor_Segmentation(preprocessed_input_image)

    ### Resistor Extraction ----------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Extract individual resistors to seperate images from the segmentation mask:
    (extracted_masks, 
     extracted_resistors,
     bounding_rects)                        = sub.Resistor_Extraction(segmentation_mask, preprocessed_input_image, min_resistor_res, target_resistor_width)
    
    ### Prepare Output Image ---------------------------------------------------------------------------------------------------------------------------------------------
    output_image = preprocessed_input_image.copy()

    ### Loop Through Extracted Objects -----------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Loop through each extracted objects, determine if they are resistors and decode the resistances if possible:
    for i in range(len(extracted_resistors)):
        mask = extracted_masks[i]
        resistor = extracted_resistors[i]
        bounding_rect = bounding_rects[i]
    
        ### Refine Resistor Mask -----------------------------------------------------------------------------------------------------------------------------------------
        # ╰┈➤ Refine extracted resistor mask using morphological operations to patch holes and (hopefully) form a contiguous complete mask:   
        refined_mask                        = sub.Refine_Resistor_Mask(mask)
        resistor = cv2.bitwise_and(resistor, resistor, mask = refined_mask)
    
        ### Inner Resistor Extraction ------------------------------------------------------------------------------------------------------------------------------------
        # ╰┈➤ Extract the inner portion of the resistor containing the bands and body:
        inner_resistor                      = sub.Inner_Resistor_Extraction(resistor, refined_mask, target_resistor_width)
    
        if (inner_resistor is None): 
            func.Display_Bounding(output_image, bounding_rect, (60,0,0), 4)
            continue

        ### Band Segmentation --------------------------------------------------------------------------------------------------------------------------------------------
        # ╰┈➤ Segment the resistor bands from the resistor body: 
        (band_mask, bodycolour_abs_diff)    = sub.Band_Segmentation(inner_resistor)
    
        inner_resistor_segmented_bands = cv2.bitwise_and(inner_resistor, inner_resistor, mask = band_mask)
        
        ### Band Extraction ----------------------------------------------------------------------------------------------------------------------------------------------
        # ╰┈➤ Extract the individual bands from inner resistor cutout using the band mask:
        (extracted_bands, 
         bands_preview, 
         largest_separation_index)          = sub.Band_Extraction(band_mask, inner_resistor, bodycolour_abs_diff, target_band_shape, object_index=i)
    
        if (extracted_bands is None):
            func.Display_Bounding(output_image, bounding_rect, (140,0,0), 4)
            continue
        
        if (bExportBandsForTraining):
            print("\n\n")
            sub.Export_Bands_For_Training(extracted_bands, target_band_shape, "training", input_index, object_index=i)
            continue
        
        ### Band Colour Classification -----------------------------------------------------------------------------------------------------------------------------------
        # ╰┈➤ Classify the colours of the extracted bands using the trained CNN model:
        extracted_band_classes              = sub.Band_Colour_Classification_CNN(extracted_bands, Colour_Classification_CNN)
        
        print("\nIndex: {} - Bands classified as: ".format(i))  
        print(extracted_band_classes)
        
        ### Decode Resistance --------------------------------------------------------------------------------------------------------------------------------------------
        # ╰┈➤ Decode the resistance and tolerance value of the resistor using the extracted band classifications:
        (decoded_resistance_string, 
         decoded_tolerance, 
         bInvertReadDirection, 
         bAmbiguousReadDirection)           = sub.Decode_Resistance(extracted_band_classes, largest_separation_index, tolerance_lookup)
    
        if (decoded_resistance_string is None):
            func.Display_Bounding(output_image, bounding_rect, (230,0,0), 4)
            continue
    
        if (len(decoded_resistance_string) == 1) or (bInvertReadDirection == False):
            display_string = decoded_resistance_string[0] + " ~" + decoded_tolerance[0]
        elif (bInvertReadDirection == True):
            display_string = decoded_resistance_string[1] + " ~" + decoded_tolerance[1]
        if (bAmbiguousReadDirection == True):
            display_string += " [*]"

        sub.Display_Resistance(output_image, display_string, bounding_rect) 
        print("\nIndex: {} Decoded Resistance: {}".format(i, display_string))
    
    ### Display Output Image ----------------------------------------------------------------------------------------------------------------------------------------------
    func.displayImage("Resistor Decoder", output_image, max_display_res)
    
    ### Wait for User Input -----------------------------------------------------------------------------------------------------------------------------------------------

    while (True):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('.'):  # Right arrow key
            input_index += 1
            if (input_index > num_input_images): input_index = 0
            status_message = "Skipping to next image..."
            break
        elif key == ord(','):  # Left arrow key
            input_index -= 1
            if (input_index < 0): input_index = num_input_images
            status_message = "Skipping to previous image..."
            break
        elif key == 27:  # Esc key
            bQuit = True
            status_message = "Quitting..."
            break
        
    status = preprocessed_input_image.copy()
    cv2.putText(status, status_message, (35, 75), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.5, color=(0, 0, 255), thickness=8, lineType=cv2.LINE_AA)
    func.displayImage("Resistor Decoder", status, max_display_res)
    cv2.waitKey(1)  # Force window update
    
    if (bQuit): break
    
    print("\n----------------------------------------------------------\n")

    

# closing all open windows 
cv2.destroyAllWindows() 