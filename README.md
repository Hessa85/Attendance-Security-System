# Attendance-Security-System 

#Creat folder for our system and uploading the images and all needed files to be in one folder to avoid any issues with path
!mkdir Attendance_Security_System

#importing all needed packages and classes which it will help to code our system
from keras import datasets
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as im

# reading and loading the images 
img = im.imread("Attendance_Security_System/Jack1.jpg")
plt.imshow(img)
plt.show()
# get dimensions of image
dimensions = img.shape
 
# height, width, number of channels in image
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
 
print('Image Dimension    : ',dimensions)
print('Image Height       : ',height)
print('Image Width        : ',width)
print('Number of Channels : ',channels)
# resizing the images because they are in different sizes and features(colored and gray)

#face detection

fd = cv2.CascadeClassifier("Attendance_Security_System/haarcascade_frontalface_alt.xml")
def get_face(img):
    if len(img.shape) ==3:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    corners = fd.detectMultiScale(img,1.3,4)
    if len(corners)==0:
        return None,None
    else:
        (x,y,w,h) = corners[0]
        img = img[y-20:y+h+20,x-20:x+w+20] # cropping the image
        img = cv2.resize(img,(100,100))
        return img
        
# loading the images from the file

import os
folder = "Attendance_Security_System"
files= os.listdir(folder)
#subfolder = os.listdir(folder)
trainimg = []
trainlb = []
def loadimage (path):
  img= im.imread(path,"PNG")
  return img
for filename in files:
  if "xml" not in filename:
    image= loadimage(folder + '/' + filename)
    img= get_face (image)
    if corner != None:
      trainimg.append (img)
      trainlb.append (filename [0])
  print (filename) #Printing all the files inside the folders
  
  def find_face(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # BGR to RGB
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j,0]>140 and img[i,j,1]<150 and img[i,j,2]<150:
                img[i,j,:]=255
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    return img
 # building the model using Keras

from keras import backend
backend.set_image_data_format("channels_first")

# Building the model
from keras import models,layers
model = models.Sequential()
# add first convolutional and maxpooling layer
model.add(layers.Conv2D(filters=20,kernel_size=(3,3),activation = 'relu',
                        input_shape = (1,100,100)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
# add the second convolutional layer
model.add(layers.Conv2D(filters=40,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
#add the flatten layer
model.add(layers.Flatten())
model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',
             metrics=['accuracy'])
          
# train the model
model.fit(trainimg,trainlb,epochs=20,batch_size=5,shuffle=True,
          verbose=True,validation_split=0.2)
         
# preprocessing the images
# scale the images
trainimg = trainimg/255

# reshape the image data
trainimg = trainimg.reshape(12,100,100)

# onehot encode the labels
from sklearn.preprocessing import OneHotEncoder
trainlb = OneHotEncoder().fit_transform(trainlb.reshape(12,1)).toarray()

print(trainimg.shape)
print(trainlb.shape)

#identifying the labels
labels = ["Happy","Normal","Sad"]

#capturing the faces 
vid = cv2.VideoCapture(0)
while True:
    ret,img = vid.read()
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corner,img2 = get_face(img2)
    if corner!=None:
        (x,y,w,h)=corner
        output = model.predict_classes(img2.reshape(1,1,100,100))
        emotion = labels[output[0]]
        cv2.putText(img,emotion,(x,y),cv2.FONT_HERSHEY_COMPLEX,
                    1.0,(0,0,255),2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),3)
    cv2.imshow("img",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
