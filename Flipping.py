import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import image
import os


def left_right_flip(img):
    backward_image = img.transpose(Image.FLIP_LEFT_RIGHT)
    return backward_image


def up_down_flip(img):
    backward_image = img.transpose(Image.FLIP_TOP_BOTTOM)
    return backward_image

Address= 'Seam Detection Project/Cropped Dataset/Training Dataset/Seam 128 Grey'
loaded_images = list()
for filename in os.listdir(Address):
    # load image
    img_data = Image.open(Address + '/'+filename)
    # store loaded image
    loaded_images.append(img_data)
   # print('> loaded %s %s' % (filename, img_data.shape))

flipped_images = list()
for img in loaded_images:
    img_ud = up_down_flip(img)
    img_lr = left_right_flip(img)
    flipped_images.extend((img_ud, img_lr))

count = 0
for img in flipped_images:
    img.save(Address + '/' + 'img_grey_flipped'+str(count)+'.jpg')
    count = count+1
