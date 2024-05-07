import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Define the constants for the model architecture and training parameters
NUM_CLASSES = 2
IMAGE_SIZE = (224, 224)
TRAIN_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 32
EPOCHS = 10

# Define the CNN architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Define the optimizer, loss function, and evaluation metric for the model
optimizer = Adam(lr=0.001)
loss_fn = categorical_crossentropy
metrics = ['accuracy']

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# Define the data generators for the training and validation sets
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'train_data',
    target_size=IMAGE_SIZE,
    batch_size=TRAIN_BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'validation_data',
    target_size=IMAGE_SIZE,
    batch_size=VALIDATION_BATCH_SIZE,
    class_mode='categorical'
)

# Define the model checkpoint, learning rate reduction, and early stopping callbacks
checkpoint_cb = ModelCheckpoint('model.h5', save_best_only=True)
reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.0001)
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=5)

# Train the model with the data generators and callbacks
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint_cb, reduce_lr_cb, early_stopping_cb]
)