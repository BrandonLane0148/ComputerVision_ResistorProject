import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy

import Functions as func

###########################################################
### Parameters and Directories
num_classes = 12        # Number of classes (resistor band colours)
#num_classes = 7
num_filters = 64        # Number of filters in the convolutional layer
filter_size = (3, 3)    # Size of the filter in the convolutional layer
pooling_size = (2, 2)   # Size of the pooling layer

num_dense_neurons = 512  # Number of neurons in the dense layer
dropout = 0.12            # Dropout rate

learn_rate = 0.00001      # Learning rate

training_data_dir = 'data_training/training/' # Root directory of the training data

num_epochs = 150  # Number of training epochs
batch_sz = 32    # Batch size

model_path = 'models/Colour_Classification_CNN.keras'


###########################################################
### Model and Layer Definitions
model = Sequential()

# Convolutional layer -> Pooling Layer
model.add(Conv2D(num_filters, filter_size, padding='same', activation='relu', input_shape=(30, 12, 3)))
model.add(MaxPooling2D(pool_size=pooling_size))

# Additional Convolutional layer
model.add(Conv2D(num_filters*2, filter_size, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=pooling_size))

model.add(Flatten())                                    # Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(num_dense_neurons, activation='relu'))  # Dense layer, aka fully connected layer. Parameters: number of neurons, activation function
model.add(Dropout(dropout))                             # Dropout layer to avoid overfitting
model.add(Dense(num_classes, activation='softmax'))     # Output layer with n neurons (for n classes) with softmax activation function (for multi-class classification)

model.compile(optimizer=Adam(learning_rate=learn_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
print('\n\n')
print('-------------------------------------------------------')

###########################################################
### Training Data Aquisition

def Preprocess_Random_Gamma(img):
    gamma = np.random.uniform(0.85, 1.15)
    img_gamma_correct = func.applyGammaCorrection(img, gamma)
    return img_gamma_correct.astype(np.uint8)

# Data augmentation
training_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.25)
    #preprocessing_function=Preprocess_Random_Gamma)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

# Fits the model on batches with real-time data augmentation
training_set = training_datagen.flow_from_directory(
    training_data_dir,
    subset='training',
    target_size=(30, 12),
    batch_size=batch_sz,
    class_mode='sparse',  # Changed class_mode to 'categorical'
    shuffle=True,
    seed=172)  

validation_set = validation_datagen.flow_from_directory(
    training_data_dir,
    subset='validation',
    target_size=(30, 12),
    batch_size=batch_sz,
    class_mode='sparse',  # Changed class_mode to 'categorical'
    shuffle=True,
    seed=172)

# Calculate the number of training images per class
#num_images_per_class = [len(os.listdir(os.path.join(training_data_dir, class_name))) for class_name in sorted(os.listdir(training_data_dir))]

# Calculate the total number of training images
#total_num_images = sum(num_images_per_class)

# Calculate the initial weights based on the number of training images per class
#class_weights = {class_index: total_num_images / (num_images_per_class[class_index] * num_classes) for class_index in range(num_classes)}

###########################################################
### Model Training
model.fit(training_set, validation_data=validation_set, epochs=num_epochs)

print('\n\n')
print('-------------------------------------------------------')
print('Model Training Complete')
print('-------------------------------------------------------')
print('\n\n')

###########################################################
### Model Validation

loss, accuracy = model.evaluate(validation_set)
print('FINAL Validation loss:', loss)
print('FINAL Validation accuracy:', accuracy)

###########################################################
### Model Export
model.save(model_path)