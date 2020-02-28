# Seam-Detection
This is a binary classification project built from the ground up. The repo includes the original manually created dataset, the augmented 
dataset, and code snippets for data augmentation, the used CNN model in tensorflow, model training, and prediction. 
### Problem Overview
This was a mini project I did for a bottle printing company that were interested in automatically detecting double sided vertical seams 
on glass containers.
The soluion was simply to emit a laser from one side of the bottle and capture the pattern of the beam on the other
side of the bottle while the bottle rotates around its axis of symmetry.  

Seam Laser Pattern| Regular Laser Pattern 
------------ | -------------
![GitHub Logo](https://github.com/4ri4Slr/Seam-Detection/blob/master/Demo%20images/photo_166.jpg)| ![GitHub Logo](https://github.com/4ri4Slr/Seam-Detection/blob/master/Demo%20images/photo_180.jpg)

### Dataset Creation and Augmentation
The original dataset consisted of 100 images manually taken and then cropped to 128*128 grayscaled images. Data augmentation was then carried out 
by flipping and changing the contrast/brightness levels of the images.

### Training 
The model includes 2 convolutinal layers each having 64 3*3 filters combined with max pooling, followed by 2 fully connected layers. Adam optimizer and binary cross entropy loss were used
to train a split of 80% - 20% traing and validation set in 50 epochs.Tensorflow 2 was used to create the model.
