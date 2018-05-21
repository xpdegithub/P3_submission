
import csv
import cv2
from keras.models import Sequential, Model
from keras.layers import Lambda,Cropping2D,Flatten,Dense,Convolution2D,ELU,Dropout,Activation
import numpy as np
import matplotlib.pyplot as plt

# Read images from left,center and right cameras
images=[]
steering_angles=[]
lines=[]
with open("carnd_p3_train/driving_log.csv") as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)
        
for line in lines:
    # create steering angles for center,left,right and flipped images
    #steering angles for images from center camera
    steering_center = float(line[3])
    steering_angles.append(steering_center)
    steering_angles.append(-steering_center)
    
    correction = 0.1 # parameter for corretion
    
    #steering angles for images from left camera
    steering_left = steering_center + correction
    steering_angles.append(steering_left)
#    steering_angles.append(-steering_left)
    
    #steering angles for images from right camera
    steering_right = steering_center - correction
    steering_angles.append(steering_right)
 #   steering_angles.append(-steering_right)
    
               
    # create image dataset. Read in images from center, left and right cameras
    source_path_0=line[0]
    filename_0=source_path_0.split("/")[-1]
    source_path_1=line[1]
    filename_1=source_path_1.split("/")[-1]
    source_path_2=line[2]
    filename_2=source_path_2.split("/")[-1]

    # read current path and create images dataset
    path="carnd_p3_train/IMG/"
    img_center = cv2.imread(path + filename_0)
    images.append(img_center)
    img_center_flipped=np.fliplr(img_center) # Augment images : Flip
    images.append(img_center_flipped)
    img_left = cv2.imread(path + filename_1)
    images.append(img_left)
    img_left_flipped=np.fliplr(img_left) # Augment images : Flip
#    images.append(img_left_flipped)
    img_right = cv2.imread(path + filename_2)
    images.append(img_right)
    img_right_flipped=np.fliplr(img_right) # Augment images : Flip
#    images.append(img_right_flipped)



X_train=np.array(images)
y_train=np.array(steering_angles)

# setup model structure                           
model = Sequential()

# Normalization use lambda layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# Crop images to ignore disturbing information
model.add(Cropping2D(cropping=((70,25), (0,0))))
# 3 convolution layers
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same",activation='relu'))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same",activation='relu'))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same",activation='relu'))
# Flatten layer
model.add(Flatten())
# Dropout
model.add(Dropout(.2))
# Elu activation layer
model.add(Activation('relu'))
# Fully connected layer
model.add(Dense(512))
# Dropout
model.add(Dropout(.5))
# Elu activation layer
model.add(Activation('relu'))
# Fully connected layer
model.add(Dense(128))
# Dropout
model.add(Dropout(.5))
# Elu activation layer
model.add(Activation('relu'))
# Fully connected layer
model.add(Dense(1))


# configure the learning process
model.compile(optimizer="adam", loss="mse")
# train the model
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=10,batch_size=128, verbose=1)

#save model as a .h5 file
model.save('model.h5')

