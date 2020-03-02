import tensorflow as tf
import os
import cv2
from tqdm import tqdm
import random
import pickle
import numpy as np

DATADIR= 'Seam Detection Project/Cropped Dataset/Training Dataset'

CATEGORIES = ['No Seam 128 Grey', 'Seam 128 Grey']

training_data = []

def create_training_data():
    for category in CATEGORIES:  #

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                training_data.append([img_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass


create_training_data()
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


print(X[1])

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
