import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
import cv2

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X = np.asarray(X)/255.0
X = np.expand_dims(X, axis=3)
y = np.asarray(y)
model = Sequential()


model.add(Conv2D(64, (3, 3), input_shape=(128, 128, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',  optimizer='adam', metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=50, validation_split=0.2)

model.save('E:\\Studies\\M.Sc\\Seam Detection Project\\model')