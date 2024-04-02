import os
from glob import glob
import random
import time
import tensorflow as tf
import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Image
import matplotlib.image as mpimg
import cv2

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint, EarlyStopping

dataset = pd.read_csv(r'C:\Users\Lenovo\state-farm-distracted-driver-detection\driver_imgs_list.csv')
# Groupby subjects
by_drivers = dataset.groupby('subject')
unique_drivers = by_drivers.groups.keys() # drivers id
NUMBER_CLASSES = 10

def get_cv2_image(path, img_rows, img_cols, color_type=3):
    """
    Function that returns an opencv image from the path and the right number of dimensions
    """
    if color_type == 1: # Loading as Grayscale image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif color_type == 3: # Loading as color image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_rows, img_cols)) # Reduce size
    return img

def load_train(img_rows, img_cols, color_type=3):
    """
    Return train images and train labels from the original path
    """
    train_images = []
    train_labels = []
    for classed in tqdm(range(NUMBER_CLASSES)):
        print('Loading directory c{}'.format(classed))
        files = glob(os.path.join(r'C:\Users\Lenovo\Downloads\imgs\train\c' + str(classed), '*.jpg'))
        for file in files:
            img = get_cv2_image(file, img_rows, img_cols, color_type)
            train_images.append(img)
            train_labels.append(classed)
    return train_images, train_labels


def read_and_normalize_train_data(img_rows, img_cols, color_type):
    """
    Load + categorical + split
    """
    X, labels = load_train(img_rows, img_cols, color_type)
    y = to_categorical(labels, 10) #categorical train label
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # split into train and test
    x_train = np.array(x_train, dtype=np.uint8).reshape(-1,img_rows,img_cols,color_type)
    x_test = np.array(x_test, dtype=np.uint8).reshape(-1,img_rows,img_cols,color_type)
    return x_train, x_test, y_train, y_test

def load_test(size=200000, img_rows=64, img_cols=64, color_type=3):
    """
    Same as above but for validation dataset
    """
    path = r'C:\Users\Lenovo\Downloads\imgs\test\*.jpg'
    files = sorted(glob(path))
    X_test, X_test_id = [], []
    total = 0
    files_size = len(files)
    for file in tqdm(files):
        if total >= size or total >= files_size:
            break
        file_base = os.path.basename(file)
        img = get_cv2_image(file, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(file_base)
        total += 1
    return X_test, X_test_id

def read_and_normalize_sampled_test_data(size, img_rows, img_cols, color_type=3):
    test_data, test_ids = load_test(size, img_rows, img_cols, color_type)
    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(-1,img_rows,img_cols,color_type)
    return test_data, test_ids

test_files, test_targets = read_and_normalize_sampled_test_data(nb_test_samples, img_rows, img_cols, color_type)

# Statistics
names = [item[17:19] for item in sorted(glob("/content/imgs/imgs/train/*"))]
activity_map = {'c0': 'Safe driving', 'c1': 'Texting - right', 'c2': 'Talking on the phone - right',
                'c3': 'Texting - left', 'c4': 'Talking on the phone - left', 'c5': 'Operating the radio',
                'c6': 'Drinking', 'c7': 'Reaching behind', 'c8': 'Hair and makeup',
                'c9': 'Talking to passenger'}

plt.figure(figsize = (12, 20))
image_count = 1
BASE_URL = r'C:\Users\Lenovo\Downloads\imgs\train'

for directory in os.listdir(BASE_URL):
    if directory[0] != '.':
        for i, file in enumerate(os.listdir(os.path.join(BASE_URL, directory))):
            if i == 1:
                break
            else:
                fig = plt.subplot(5, 2, image_count)
                image_count += 1
                image_path = os.path.join(BASE_URL, directory, file)
                image = mpimg.imread(image_path)

                plt.imshow(image)
                plt.title(activity_map[directory])

x_train, x_test, y_train, y_test = read_and_normalize_train_data(img_rows, img_cols, color_type) 
#Number of batch size and epochs
batch_size = 32
nb_epoch = 20

checkpointer = ModelCheckpoint(filepath='saved_models/weights_best_vanilla.keras', 
                               monitor='val_loss', mode='min',
                               verbose=1, save_best_only=True)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import Dropout

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras import backend as K

def create_model(img_rows, img_cols, color_type, NUMBER_CLASSES):
    if color_type == 1:
        input_shape = (img_rows, img_cols, 3)  # Convert grayscale to RGB
    else:
        input_shape = (img_rows, img_cols, color_type)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(NUMBER_CLASSES, activation='softmax'))

    return model
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_input_images(images, color_type):
    processed_images = []
    for img in images:
        if color_type == 1:
            # Convert grayscale to RGB by stacking the single channel three times
            img_rgb = np.stack((img,)*3, axis=-1)
            processed_images.append(img_rgb)
        else:
            # If images are already RGB, keep them unchanged
            processed_images.append(img)
    return np.array(processed_images)

# Preprocess the input images
x_train_processed = preprocess_input_images(x_train_processed, color_type)
x_test_processed = preprocess_input_images(x_test_processed, color_type)


model = create_model(img_rows, img_cols, color_type, NUMBER_CLASSES)


#model = create_model(img_rows, img_cols, color_type, NUMBER_CLASSES)
# Compiling the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=nb_epoch, batch_size=batch_size,
                    callbacks=[checkpointer, es], verbose=1)

# Plotting training history
def plot_train_history(history):
    """
    Plot the validation accuracy and validation loss over epochs
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

plot_train_history(history)

# Using ImageDataGenerator from keras
# Data Augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Add validation split if you want
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

# Flow training images in batches of batch_size using train_datagen generator
train_generator = train_datagen.flow(
    x_train,
    y_train,
    batch_size=batch_size,
    subset='training'  # Indicate the subset for training data
)

# Flow validation images in batches of batch_size using train_datagen generator
validation_generator = train_datagen.flow(
    x_train,
    y_train,
    batch_size=batch_size,
    subset='validation'  # Indicate the subset for validation data
)

# Fit the model
history_v2 = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=nb_epoch,
    verbose=1,
    callbacks=[checkpointer, es],
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Saving the model
model.save('cnnmodel.h5')