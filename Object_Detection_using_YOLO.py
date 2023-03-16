#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries 
import cv2 # Image Processing Lib
from keras.models import load_model # To used Pretrained Model 
from PIL import Image, ImageOps # To Modifie the INput image 
import numpy as np # To convert Input Imgage into NUmpy Array To send Furthur Process 
# from tracker import *


# In[ ]:


from object_detection import ObjectDetection # The object dectedction File is External 
import math # for Calculation 


# In[ ]:


od = ObjectDetection() # converting Class into an object 


# In[ ]:


cap = cv2.VideoCapture("/home/shraddha/Drone Detection/Drone detection videos/videos/ship3.mp4") # input Video 


# In[ ]:


# cap = cv2.VideoCapture(0) # Webcamp Input 


# In[ ]:


# Load the model
model = load_model('/home/shraddha/Drone Detection/converted_keras/keras_model.h5',compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


# In[ ]:


# method of measuring the execution time of your python code snippets. 
# import timeit
# mysetup="from keras.models import load_model"
# mycode='''model = load_model('/home/shraddha/Desktop/cnn/New cnn mode-20220912T090732Z-001/New cnn mode/teachable machine/code from website/keras_model.h5',compile=False)'''
# print(timeit.timeit(setup=mysetup,stmt=mycode,number=1))


# In[1]:


# Initialize count
count = 0
center_points_prev_frame = []
tracking_objects = {}
track_id = 0


# In[ ]:


def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 18
    speed = d_meters * fps * 3.6
    return speed


# # 0 ship
# # 1 helicopter
# # 2 bird
# # 3 Aeroplane
# # 4 Drone

# In[ ]:


current_frame = 0
while(True):
    ret, frame = cap.read()
    
    if ret:
        name = 'frame'+str(current_frame)+'.jpg'
        cv2.imwrite(name,frame)
        if(current_frame==5):
            current_frame=1
        else:
            current_frame+=1
    else:
        break
    
    image = Image.open(name)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    
    
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float64) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = model.predict(data)
    print(prediction)
    x = max(prediction)
    print(x)
    x = max(prediction[0,:])
    a=x*100
    print(a,'percentage')
    category = ["Ship","Helicopter","Bird","Aeroplane","Drone"]
    i=0
    while(i<4):
    
        if (prediction[0,i] == x):
            print(i)
            print(category[i])  
            break

        else:
            i=i+1
    cv2.imshow('Sucessfully detected',frame)
    
    count += 1
    if not ret:
        break

    # Point current frame
    center_points_cur_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, (str(center_points_cur_frame)), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    print("Tracking objects")
    print(tracking_objects)


    print("CUR FRAME LEFT PTS")
    print(center_points_cur_frame)
    
    for i in objectLocation1.keys():	
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = objectLocation1[i]
                [x2, y2, w2, h2] = objectLocation2[i]
            objectLocation1[i] = [x2, y2, w2, h2]
            
            if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                    speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])
                    
                    if speed[i] != None and y1 >= 180:
                        cv2.putText(frame, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.imshow('result', frame)


    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

cap.release()
cv2.destroyAllWindows()


# In[ ]:




