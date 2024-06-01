import cv2 #opencv itself
import numpy as np # matrix manipulations
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

###########################################################
### Initialisation

colour_lookup =    ["Black", "Brown", "Red" , "Orange", "Yellow", "Green", "Blue" , "Violet", "Grey" , "White", "Gold", "Silver"]

training_data_dir = 'data_training/testing/' # Root directory of the training data

# Load the trained CNN model
Colour_Classification_CNN = load_model('models/Colour_Classification_CNN.keras')

# Set up the data generator for testing dataset
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the testing dataset
test_set = test_datagen.flow_from_directory(
    training_data_dir,
    target_size=(30, 12),
    batch_size=1,
    class_mode='sparse',
    shuffle=False
)


###########################################################
### Model Validation

loss, accuracy = Colour_Classification_CNN.evaluate(test_set)
print('FINAL Validation loss:', loss)
print('FINAL Validation accuracy:', accuracy)

###########################################################

cv2.waitKey(0) # waits for user to press any key