import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
import scipy
import matplotlib.pyplot as plt

import Functions as func

###########################################################
### Parameters and Directories
num_classes = 12        # Number of classes (resistor band colours)
num_filters = 64        # Number of filters in the convolutional layer
filter_size = (3, 3)    # Size of the filter in the convolutional layer
pooling_size = (2, 2)   # Size of the pooling layer

num_dense_neurons = 512  # Number of neurons in the dense layer
dropout = 0.12            # Dropout rate

learn_rate = 0.00001      # Learning rate

training_data_dir = 'data_training/training/' # Root directory of the training data

num_epochs = 200  # Number of training epochs
batch_sz = 32     # Batch size

model_path = 'models/Colour_Classification_CNN.keras'


###########################################################
### Model Architecture
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
model.add(Dense(num_dense_neurons*2, activation='relu'))  # Dense layer, aka fully connected layer. Parameters: number of neurons, activation function
model.add(Dropout(dropout*2))                             # Dropout layer to avoid overfitting
model.add(Dense(num_classes, activation='softmax'))     # Output layer with n neurons (for n classes) with softmax activation function (for multi-class classification)

model.compile(optimizer=Adam(learning_rate=learn_rate), loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

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
    class_mode='categorical',  # Changed class_mode to 'categorical'
    shuffle=True,
    seed=142)  

validation_set = validation_datagen.flow_from_directory(
    training_data_dir,
    subset='validation',
    target_size=(30, 12),
    batch_size=batch_sz,
    class_mode='categorical',  # Changed class_mode to 'categorical'
    shuffle=True,
    seed=142)

###########################################################
### Model Training
history = model.fit(training_set, validation_data=validation_set, epochs=num_epochs)
#history = model.fit(training_set, epochs=num_epochs)  #No validation set


print('\n\n')
print('-------------------------------------------------------')
print('Model Training Complete')
print('-------------------------------------------------------')
print('\n\n')

###########################################################
### Training History Plots

# Plot training & validation accuracy values
plt.figure(figsize=(14,6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Plot training & validation precision values
plt.figure(figsize=(14,6))

plt.subplot(1, 2, 1)
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('Model Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation recall values
plt.subplot(1, 2, 2)
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('Model Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()


###########################################################
### Model Validation

loss, accuracy, precision, recall = model.evaluate(validation_set)
print('FINAL Validation loss:', loss)
print('FINAL Validation accuracy:', accuracy)
print('FINAL Validation precision:', precision)
print('FINAL Validation recall:', recall)

###########################################################
### Model Export
model.save(model_path)