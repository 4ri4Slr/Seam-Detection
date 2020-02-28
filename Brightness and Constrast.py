import cv2
import numpy as np
import os
import random

Address = 'Cropped Dataset/Training Dataset/No Seam 128 Grey'

loaded_images = list()

for filename in os.listdir(Address):
    # load image
    img_data = cv2.imread(Address + '/' + filename)
    # store loaded image
    loaded_images.append(img_data)
# print('> loaded %s %s' % (filename, img_data.shape))

alpha = round(random.uniform(1, 1.25), 3)
beta = random.randrange(-50, 25, 1)

br_con_images = list()
for img in loaded_images:
    img_br = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
    br_con_images.append(img_br)

    count = 0
    for im_data in br_con_images:
        cv2.imwrite(
            Address + '/' + 'img_br_co' + str(
                count) + '.jpg', im_data)
        count = count + 1
