import tensorflow as tf
import cv2
import glob
import numpy as np


IMG_SIZE=128

new_model = tf.keras.models.load_model('Seam Detection Project/model')
new_model.summary()


images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob('Seam Detection Project/Demo images/*.jpg')]

images = np.asarray(images)
seam_test = cv2.resize(images[0], (IMG_SIZE, IMG_SIZE))
noSeam_test = cv2.resize(seam_test[1], (IMG_SIZE, IMG_SIZE))
cv2.imshow('seam', seam_test)

seam_test = seam_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255.0
noSeam_test = noSeam_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255.0


prediction1 = new_model.predict(seam_test)
prediction0 = new_model.predict(noSeam_test)

print(prediction0, prediction1)
