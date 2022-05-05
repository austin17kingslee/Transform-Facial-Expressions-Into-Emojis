from tkinter import *
import cv2
import os
import numpy as np
import cv2
import pydot 
import PIL
from PIL import Image
from PIL import ImageTk
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model

train_dir = 'data/archive/train'
val_dir = 'data/archive/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
#training generator for CNN
train_generator = train_datagen.flow_from_directory(
       train_dir,
       target_size=(48,48),
       batch_size=64,
       color_mode="grayscale",
       class_mode='categorical')
#validation generator for CNN
validation_generator = val_datagen.flow_from_directory(
       val_dir,
       target_size=(48,48),
       batch_size=64,
       color_mode="grayscale",
       class_mode='categorical')

for i in os.listdir(train_dir):
    print(str(len(os.listdir(train_dir + "/" + i))) +" "+ i +" images")
for i in os.listdir(val_dir):
    print(str(len(os.listdir(val_dir + "/" + i))) +" "+ i +" images")

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))#output=(48-3+0)/1+1=46
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))#output=(46-3+0)/1+1=44
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))#output=devided input by 2 it means 22,22,64
emotion_model.add(Dropout(0.25))#reduce 25% module at a time of output
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',input_shape=(48,48,1)))#(22-3+0)/1+1=20
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))#10
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))#(10-3+0)/1+1=8
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))#output=4
emotion_model.add(Dropout(0.25))#nothing change
emotion_model.add(Flatten())#here we get multidimension output and pass as linear to the dense so that 4*4*128=2048
emotion_model.add(Dense(1024, activation='relu'))#hddien of 1024 neurons of input #note find out what this is 
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))#hddien of 7 neurons of input
plot_model(emotion_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)#save model leyer as model_plot.png
emotion_model.summary()

emotion_model.compile(loss='categorical_crossentropy',optimizer=adam_v2.Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
emotion_model_info = emotion_model.fit_generator( #to fetch the model info from validation generator
       train_generator,
       steps_per_epoch=28709 // 64,
       epochs=50,
       validation_data=validation_generator,
       validation_steps=7178 // 64)

emotion_model.save_weights('model.h5')#to save the model

cv2.ocl.setUseOpenCL(False)
#emotion dictionary creation
em_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
cap = cv2.VideoCapture(0)
while True:
    ret, fram = cap.read()
    if not ret:
        break
#bounding box initialization  
bounding_box = cv2.CascadeClassifier('/home/shikha/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
gray_frame = cv2.cvtColor(fram, cv2.COLOR_BGR2gray_frame)
#to detect the multiple faces and frame them separately   
n_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
for (x, y, w, h) in n_faces:
    cv2.rectangle(fram, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
    roi_frame = gray_frame[y:y + h, x:x + w]
    crop_img = np.expand_dims(np.expand_dims(cv2.resize(roi_frame, (48, 48)), -1), 0)
    emotion_prediction = emotion_model.predict(crop_img)
    maxindex = int(np.argmax(emotion_prediction))
    cv2.putText(roi_frame, em_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Video', cv2.resize(roi_frame,(1200,860),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break