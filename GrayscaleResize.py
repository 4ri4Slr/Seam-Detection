import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import image
import os
IMAGE_SIZE = 128

def resize_images(image):

    tf_img = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    tf_img=np.uint8(tf_img)
    backward_image=Image.fromarray(tf_img)
    return backward_image

loaded_images = list()


for filename in os.listdir('Seam Detection Project/Cropped Dataset/Test Dataset/Seam'):
    # load image
    img_data = image.imread('Seam Detection Project/Cropped Dataset/Test Dataset/Seam/'+filename)
    # store loaded image
    loaded_images.append(img_data)
    print('> loaded %s %s' % (filename, img_data.shape))

# Resize and GreyScale the Images
resized_images=list()
for img in loaded_images:
    img_res=resize_images(img)
    resized_images.append(img_res.convert(mode='L'))

# Save Resized Images
count=0
for img in resized_images:
    img.save('Seam Detection Project/Cropped Dataset/Test Dataset/'+ 'img_grey'+str(count)+'.jpg')
    count=count+1
