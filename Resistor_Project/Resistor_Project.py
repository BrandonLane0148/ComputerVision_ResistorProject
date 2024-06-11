
import cv2 
import numpy as np
from tensorflow.keras.models import load_model

import Functions as func
import Subroutines as sub

#################################################################################################################################################################################################
### Initialisation ##############################################################################################################################################################################     
# ╰┈➤ Initialise various parameters, lookup tables, and load the CNN model:

colour_lookup =     ["Black"   , "Brown"    , "Red"      , "Orange"    , "Yellow"    , "Green"    , "Blue"     , "Violet"    , "Grey"       , "White"      , "Gold"      , "Silver"     ]
tolerance_lookup =  ["ERR%"    , "1%"       , "2%"       , "ERR%"      , "ERR%"      , "0.5%"     , "0.25%"    , "0.1%"      , "0.05%"      , "ERR%"       , "5%"        , "10%"        ]
colour_BGR_lookup = [(60,40,40), (65,70,110), (50,40,130), (40,100,180), (70,170,170), (50,100,20), (150,60,10), (150,60,100), (130,135,135), (240,230,220), (80,170,220), (195,190,195)]


dir_generic = 'data/'
dir_stress = 'data/stress/'
current_dir = dir_generic

num_input_images = 47
num_input_images_stress = 6

max_input_res = [2560, 2560]
max_display_res = [1440, 2560]
#max_display_res = [900, 1920]

min_resistor_res = 30
target_resistor_width = 300
target_band_shape = (12, 30)

bExportBandsForTraining = False

# Load the trained CNN model
Colour_Classification_CNN = load_model('models/Colour_Classification_CNN.keras')


#################################################################################################################################################################################################
### Main Program Loop ###########################################################################################################################################################################     
# ╰┈➤ Loop through all input images, and perform the resistor decoding process:

input_index = 0
input_index_old = 0
bUseCustomImage = False

while (True):
    
    ### Load Input Image -------------------------------------------------------------------------------------------------------------------------------------------------
    
    if (bUseCustomImage):
        input_image = cv2.imread('data/import.jpg')
        print("Using Custom Image")
        bUseCustomImage = False
    else:
        input_image = cv2.imread(current_dir+str(input_index)+'.jpg')
        print("Image Index: {}".format(input_index))

    ### Preprocess Input Image -------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Preprocess the input image to resize and filter out noise:
    preprocessed_input_image                = sub.Input_Image_Preprocessing(input_image, max_input_res)

    func.displayImage("Resistor Decoder", preprocessed_input_image, max_display_res)
    
    ### Process User Input -----------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Wait for user input to change image, quit, or continue with current image:
    
    status_message = ""
    bSkip = False
    bQuit = False
    while (True):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('.'):                                                 #If the right arrow key is pressed:
            input_index += 1                                                    #Increment the input index
            if (current_dir == dir_generic) and (input_index > num_input_images): input_index = 0
            elif (current_dir == dir_stress) and (input_index > num_input_images_stress): input_index = 0
            bSkip = True
            status_message = "Skipping to next image [{}]...".format(input_index)
            break
        elif key == ord(','):                                               #If the left arrow key is pressed:
            input_index -= 1                                                    #Decrement the input index
            if (input_index < 0): 
                if (current_dir == dir_generic): input_index = num_input_images
                elif (current_dir == dir_stress): input_index = num_input_images_stress
            bSkip = True
            status_message = "Skipping to previous image [{}]...".format(input_index)
            break
        elif key == ord('r'):                                               #If the 'r' key is pressed:
            if (current_dir == dir_generic):                                    #Select random input image
                input_index = np.random.randint(0, num_input_images)
            elif (current_dir == dir_stress): 
                input_index = np.random.randint(0, num_input_images_stress)
            status_message = "Selecting random image [{}]...".format(input_index)
            bSkip = True
            break
        elif key == ord('0'):                                               #If the '0' key is pressed:
            input_index = 0                                                     #Set the input index to 0
            status_message = "Selecting first image [{}]...".format(input_index)
            bSkip = True
            break
        elif key == ord('i'):                                               #If the 'i' key is pressed:
            bUseCustomImage = True                                              #Set the custom image flag to True
            status_message = "Importing custom image..."
            bSkip = True
            break
        elif key == ord('t'):                                               #If the 't' key is pressed:
            if (current_dir == dir_generic):                                    #Swap test set
                current_dir = dir_stress
                status_message = "Switching to stress test images..."
            elif (current_dir == dir_stress): 
                current_dir = dir_generic
                status_message = "Switching to generic test images..."
            input_index, input_index_old = input_index_old, input_index
            bSkip = True
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
     segmentation_mask_boundings, 
     bounding_rects)                        = sub.Resistor_Extraction(segmentation_mask, preprocessed_input_image, min_resistor_res, target_resistor_width)
    
    ### Prepare Output Image ---------------------------------------------------------------------------------------------------------------------------------------------
    output_image = preprocessed_input_image.copy()
    object_details = []

    #############################################################################################################################################################################################
    ### Loop Through Extracted Objects ########################################################################################################################################################## 
    # ╰┈➤ Loop through each extracted objects, determine if they are resistors and decode the resistances if possible:
    for i in range(len(extracted_resistors)):
        mask = extracted_masks[i]
        resistor = extracted_resistors[i]
        bounding_rect = bounding_rects[i]
        
        object_details_log = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        preview_objectID = func.Construct_ObjectID_Preview(i, target_resistor_width)
        
        ### Re-evaluate Resistor Mask ------------------------------------------------------------------------------------------------------------------------------------
        # ╰┈➤ Perform resistor segmentation again within the confined bounding rectangle to allow more effective OTSU thresholding:
        mask                                = sub.Resistor_Segmentation(resistor, gamma_correction=2, threshold_factor=0.7)
        
        object_details_log = func.Append_To_Object_Log(mask, object_details_log, bool_convBGR=True)
    
        ### Refine Resistor Mask -----------------------------------------------------------------------------------------------------------------------------------------
        # ╰┈➤ Refine extracted resistor mask using morphological operations to patch holes and (hopefully) form a contiguous complete mask:   
        refined_mask                        = sub.Refine_Resistor_Mask(mask)
        
        resistor = cv2.bitwise_and(resistor, resistor, mask = refined_mask)
        object_details_log = func.Append_To_Object_Log(refined_mask, object_details_log, bool_convBGR=True)
        object_details_log = func.Append_To_Object_Log(resistor, object_details_log, padding=5)
    
        ### Inner Resistor Extraction ------------------------------------------------------------------------------------------------------------------------------------
        # ╰┈➤ Extract the inner portion of the resistor containing the bands and body:
        (inner_resistor, offset)            = sub.Inner_Resistor_Extraction(resistor, refined_mask, target_resistor_width)
    
        if (inner_resistor is None):                                                            #If the inner resistor extraction failed:
            func.Display_Bounding(output_image, bounding_rect, (60,0,0), 4)                         #Display bounding as incomplete colour (stage 1)
            object_details_log = func.Append_To_Object_Log(preview_objectID, object_details_log, bool_upwards=True)
            object_details.append(object_details_log)                                               #Append the object details log to the object details list
            continue                                                                                #Skip to the next object
        
        object_details_log = func.Append_To_Object_Log(inner_resistor, object_details_log, offset, padding=10)

        ### Band Segmentation --------------------------------------------------------------------------------------------------------------------------------------------
        # ╰┈➤ Segment the resistor bands from the resistor body: 
        (band_mask, bodycolour_abs_diff)    = sub.Band_Segmentation(inner_resistor)
    
        inner_resistor_segmented_bands = cv2.bitwise_and(inner_resistor, inner_resistor, mask = band_mask)
        object_details_log = func.Append_To_Object_Log(inner_resistor_segmented_bands, object_details_log, offset, padding=10)
        
        ### Band Extraction ----------------------------------------------------------------------------------------------------------------------------------------------
        # ╰┈➤ Extract the individual bands from inner resistor cutout using the band mask:
        (extracted_bands, 
         bands_preview, 
         largest_separation_index)          = sub.Band_Extraction(band_mask, inner_resistor, bodycolour_abs_diff, target_band_shape, object_index=i)
    
        if (extracted_bands is None):                                                         #If band extraction failed:  
            func.Display_Bounding(output_image, bounding_rect, (140,0,0), 4)                        #Display bounding as incomplete colour (stage 2)
            object_details_log = func.Append_To_Object_Log(preview_objectID, object_details_log, bool_upwards=True)
            object_details.append(object_details_log)                                               #Append the object details log to the object details list
            continue                                                                                #Skip to the next object
        
        if (bExportBandsForTraining):
            print("\n\n")
            sub.Export_Bands_For_Training(extracted_bands, target_band_shape, "testing", input_index, object_index=i)
        
        object_details_log = func.Append_To_Object_Log(bands_preview, object_details_log, offset, padding=20)
        
        ### Band Colour Classification -----------------------------------------------------------------------------------------------------------------------------------
        # ╰┈➤ Classify the colours of the extracted bands using the trained CNN model:
        extracted_band_classes              = sub.Band_Colour_Classification_CNN(extracted_bands, Colour_Classification_CNN)
                
        print("\nIndex: {} - Bands classified as: ".format(i))  
        print(extracted_band_classes)
        
        (preview_nums, preview_colours) = func.Construct_Classification_Preview(extracted_band_classes, target_resistor_width, colour_lookup, colour_BGR_lookup)
        object_details_log = func.Append_To_Object_Log(preview_nums, object_details_log, padding=5)
        object_details_log = func.Append_To_Object_Log(preview_colours, object_details_log, padding=12)
        
        ### Decode Resistance --------------------------------------------------------------------------------------------------------------------------------------------
        # ╰┈➤ Decode the resistance and tolerance value of the resistor using the extracted band classifications:
        (decoded_resistance_string, 
         decoded_tolerance, 
         bInvertReadDirection, 
         bAmbiguousReadDirection)           = sub.Decode_Resistance(extracted_band_classes, largest_separation_index, tolerance_lookup)
    
        if (decoded_resistance_string is None):                                             #If resistance decoding failed:
            func.Display_Bounding(output_image, bounding_rect, (230,0,0), 4)                    #Display bounding as incomplete colour (stage 3)
            object_details_log = func.Append_To_Object_Log(preview_objectID, object_details_log, bool_upwards=True)
            object_details.append(object_details_log)                                           #Append the object details log to the object details list
            continue                                                                            #Skip to the next object
    
        if (len(decoded_resistance_string) == 1) or (bInvertReadDirection == False):        
            display_string = "{} ~{}".format(decoded_resistance_string[0], decoded_tolerance[0])
        elif (bInvertReadDirection == True):
            display_string = "{} ~{}".format(decoded_resistance_string[1], decoded_tolerance[1])
        if (bAmbiguousReadDirection == True):
            display_string += " [*]"

        sub.Display_Resistance(output_image, display_string, bounding_rect) 
        print("\nIndex: {} Decoded Resistance: {}".format(i, display_string))
        
        if (len(decoded_resistance_string) == 2):
            if (bInvertReadDirection): display_string = "{} ~{}".format(decoded_resistance_string[0], decoded_tolerance[0])
            else: display_string = "{} ~{}".format(decoded_resistance_string[1], decoded_tolerance[1])
            preview_inverted_resistance = func.Construct_Inverted_Resistance_Preview(display_string, bAmbiguousReadDirection, target_resistor_width)
            object_details_log = func.Append_To_Object_Log(preview_inverted_resistance, object_details_log, padding=5, bool_upwards=True)
            
        object_details_log = func.Append_To_Object_Log(preview_objectID, object_details_log, bool_upwards=True)
        object_details.append(object_details_log)
        
        status = output_image.copy()
        cv2.putText(status, status_message, (35, 75), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.5, color=(0, 0, 255), thickness=8, lineType=cv2.LINE_AA)
        func.displayImage("Resistor Decoder", status, max_display_res)
        cv2.waitKey(1)  # Force window update
        
    #############################################################################################################################################################################################
    ### Display Output ##########################################################################################################################################################################
   
    func.displayImage("Resistor Decoder", output_image, max_display_res)
    
    ### Process User Input -----------------------------------------------------------------------------------------------------------------------------------------------
    # ╰┈➤ Wait for user input to change image or quit:
    bDisplaySegmentation = False
    while (True):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('.'):                                                 #If the right arrow key is pressed:
            input_index += 1                                                    #Increment the input index
            if (current_dir == dir_generic) and (input_index > num_input_images): input_index = 0
            elif (current_dir == dir_stress) and (input_index > num_input_images_stress): input_index = 0
            status_message = "Changing to next image [{}]...".format(input_index)
            break
        elif key == ord(','):                                               #If the left arrow key is pressed:
            input_index -= 1                                                    #Decrement the input index
            if (input_index < 0): 
                if (current_dir == dir_generic): input_index = num_input_images
                elif (current_dir == dir_stress): input_index = num_input_images_stress
            status_message = "Changing to previous image [{}]...".format(input_index)
            break
        elif key == ord('r'):                                               #If the 'r' key is pressed:
            if (current_dir == dir_generic):                                    #Select random input image
                input_index = np.random.randint(0, num_input_images)
            elif (current_dir == dir_stress): 
                input_index = np.random.randint(0, num_input_images_stress)
            status_message = "Selecting random image [{}]...".format(input_index)
            break
        elif key == ord('0'):                                               #If the '0' key is pressed:
            input_index = 0                                                     #Set the input index to 0
            status_message = "Selecting first image [{}]...".format(input_index)
            break
        elif key == ord('i'):                                               #If the 'i' key is pressed:
            bUseCustomImage = True                                              #Set the custom image flag to True
            status_message = "Importing custom image..."
            break
        elif key == ord('t'):                                               #If the 't' key is pressed:
            if (current_dir == dir_generic):                                    #Swap test set
                current_dir = dir_stress
                status_message = "Switching to stress test images..."
            elif (current_dir == dir_stress): 
                current_dir = dir_generic
                status_message = "Switching to generic test images..."
            input_index, input_index_old = input_index_old, input_index
            break
        elif key == ord('s'):                                               #If the 's' key is pressed:                        
            if (bDisplaySegmentation == False):                                 #Display the segmentation mask if it's not already displayed
                bDisplaySegmentation = True
                func.displayImage("Resistor Decoder", segmentation_mask_boundings, max_display_res)
            else:
                bDisplaySegmentation = False                                    #Otherwise revert to the original output image if it is
                func.displayImage("Resistor Decoder", output_image, max_display_res)
        elif key == 27:                                                     #If the escape key is pressed:
            bQuit = True                                                        #Set the quit flag to True
            status_message = "Quitting..."
            break
        
        (MouseX, MouseY) = func.QueryMouseClick()                           #Query the mouse status
        MousePos = (MouseX, MouseY)
        if (MouseX is not None) and (MouseY is not None):                   #If a left click has occured:
            for i in range(len(bounding_rects)):                                #Iterate through the bounding rectangles:
                rect = bounding_rects[i]
                box = cv2.boxPoints(rect)                                           #Obtain box points from bounding rectangle
                box = np.int0(box)  
                
                result = cv2.pointPolygonTest(box, MousePos, False)                 #Check if the mouse click is within the box points

                if (result >= 0):                                                   #If the mouse click is within the box points:
                    cv2.imshow("Object Details", object_details[i])                     #Display the respective object details panel
                    break
                  
    status = preprocessed_input_image.copy()
    cv2.putText(status, status_message, (35, 75), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.5, color=(0, 0, 255), thickness=8, lineType=cv2.LINE_AA)
    func.displayImage("Resistor Decoder", status, max_display_res)
    cv2.waitKey(1)  # Force window update
    
    if (bQuit): break
    
    print("\n----------------------------------------------------------\n")

    

# closing all open windows 
cv2.destroyAllWindows() 