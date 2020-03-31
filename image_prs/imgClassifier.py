#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load images from local directories

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "images"
CATEGORIES = ["controls", "cases"]
IMG_SIZE = 12

training_data = []
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
        except Exception as e:
            pass

print(len(training_data))
# fig, ax = plt.subplots()
# ax.imshow(training_data[0][0], cmap = 'gray')
# plt.show()

# In[7]:


# Generate training data and labels

import random
random.shuffle(training_data)

X = []
y = []

count = 0

for features, label in training_data:
    if label == 0 and count < 5000:
        X.append(features)
        y.append(label)
        count = count + 1
    elif label == 1:
        for i in range(1):
            X.append(features)
            y.append(label)

print(len(X))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[8]:


# Build the CNN and evaluate on training data

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

NAME = "Colon-CNN-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

X = X / 255.0

xSorted = X.tolist()
ySorted = y[:]

xSorted = [x for _,x in sorted(zip(ySorted, xSorted))]
ySorted = sorted(y)

xTrain = xSorted[:5400]
xTrain = np.array(xTrain).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
yTrain = ySorted[:5400]
xTest = xSorted[5401:]
xTest = np.array(xTest).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
yTest = ySorted[5401:]

# In[9]:


model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
# model.add(Dropout(0.25))

for i in range(2):
    model.add(Dense(128))
    model.add(Activation("relu"))

model.add(Dropout(0.2))

for i in range(1):
    model.add(Dense(64))
    model.add(Activation("relu"))

model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation("sigmoid"))

# model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy", tf.keras.metrics.AUC()])

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives()])
output = model.fit(X, y, batch_size = 10, epochs = 20, validation_split = 0.2, callbacks = [tensorboard])

# print(output.history.keys())


# # In[6]:


# # Calculate and print evaluation metrics
# TP = float(output.history["true_positives"][-1])
# TN = float(output.history["true_negatives"][-1])
# FP = float(output.history["false_positives"][-1])
# FN = float(output.history["false_negatives"][-1])

# sensitivity = TP / (TP + FN)
# specificity = TN / (TN + FP)

# LD_Pos = sensitivity / (1 - specificity)
# LD_Neg = (1 - sensitivity) / specificity

# print("LD_Pos: " + str(LD_Pos))
# print("LD_Neg: " + str(LD_Neg))


# In[ ]:




