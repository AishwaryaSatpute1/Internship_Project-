#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from object_detection import ObjectDetection
import math
from kalmanfilter import KalmanFilter
from orange_detector import OrangeDetector


# In[11]:


od = ObjectDetection()


# In[12]:


kf = KalmanFilter()


# In[13]:


cap = cv2.VideoCapture("/home/shraddha/Drone Detection/Drone detection videos/MP4drone_video-20220921T054401Z-001/MP4drone_video/v3.mp4")


# In[14]:



# Load the model
model = load_model('/home/shraddha/Desktop/cnn/New cnn mode-20220912T090732Z-001/New cnn mode/teachable machine/code from website/keras_model.h5',compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


# In[15]:


# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0


# In[16]:


current_frame = 0
while(True):
    ret, frame = cap.read()
    cv2.waitKey(10)
    
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
        #print("FRAME N°", count, " ", x, y, w, h)
    

        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        predicted = kf.predict(cx, cy)
    #cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 4)
#         cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 4)
        cv2.circle(frame, (predicted[0], predicted[1]),20, (255, 0, 0), 4)

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

        cv2.putText(frame, (str(int(a))+ "%" + category[i]), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    print("Tracking objects")
    print(tracking_objects)


    print("CUR FRAME LEFT PTS")
    print(center_points_cur_frame)
    
    
class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        print(x,y)
        return x, y

    predicted = kf.predict(cx, cy)
    cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 4)
    cv2.circle(frame, (predicted[0], predicted[1]), 20, (255, 0, 0), 4)


    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




