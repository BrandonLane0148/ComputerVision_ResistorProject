import cv2 #opencv itself
import numpy as np # matrix manipulations
from tensorflow.keras.models import load_model

###########################################################
### Initialisation

colour_lookup =    ["Black", "Brown", "Red" , "Orange", "Yellow", "Green", "Blue" , "Violet", "Grey" , "White", "Gold", "Silver"]

# Load the trained CNN model
Colour_Classification_CNN = load_model('models/Colour_Classification_CNN.keras')

band = cv2.imread('data_bands/training/5/15_2_4.jpg')

band_RGB = cv2.cvtColor(band, cv2.COLOR_BGR2RGB)    # CNN is trained using RGB images
band_RGB = band_RGB / 255.0                         # CNN is trained using images with pixel values between 0 and 1
band = band.reshape(1, 30, 12, 3)                   # CNN expects batch size, height, width, channels
band_predictions = Colour_Classification_CNN.predict(band, verbose=0)                # Use the CNN model to predict the class of the band
band_class = np.argmax(band_predictions)            # Classify as the class with the highest probability
if (band_class == 6): band_class = 10 #TEMP

print(str(band_class) + ': ' + colour_lookup[int(band_class)])

cv2.waitKey(0) # waits for user to press any key 