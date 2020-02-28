import tensorflow as tf
import cv2

IMG_SIZE=128

new_model = tf.keras.models.load_model('E:\\Studies\\M.Sc\\Seam Detection Project\\model')
new_model.summary()

seam_test = cv2.imread('E:\\Studies\\M.Sc\\Seam Detection Project\\Demo images\\photo_166.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Seam", seam_test)

noSeam_test = cv2.imread('E:\\Studies\\M.Sc\\Seam Detection Project\\Demo images\\photo_180.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Seam", noSeam_test)

seam_test = cv2.resize(seam_test, (IMG_SIZE, IMG_SIZE))
noSeam_test = cv2.resize(noSeam_test, (IMG_SIZE, IMG_SIZE))

seam_test = seam_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255.0
noSeam_test = noSeam_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255.0

prediction1 = new_model.predict(seam_test)
prediction0 = new_model.predict(noSeam_test)

print(prediction0, prediction1)