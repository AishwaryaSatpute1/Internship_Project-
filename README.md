# Internship_Project
This Project Was Part of Internship - Aerial Vehicle Detection and tracking 


## Demo

Insert gif or link to demo

![](https://github.com/AishwaryaSatpute1/Internship_Project-/blob/main/KF_aeroplane_result.gif)

## Documentation
We show examples on how to perform the following parts of the Deep Learning workflow:

## 1. Collecting  and labeling data: 
Gathered images and labelled them using a tool  Roboflow's built-in image annotation tool 
[Aeroplan Dataset](https://universe.roboflow.com/enes-demirtas/rota_yolov4)

[Drone  Dataset](https://universe.roboflow.com/drone-rwsrk/drone-cmxwz)

[Helicopter Dataset](https://universe.roboflow.com/ahmedmohsen/helicopter-s9vfb)

[Bird Dataset](https://universe.roboflow.com/eng-word/bird-mv9br)

[Ship Dataset](https://universe.roboflow.com/wilson_xu_weixuan-outlook-com/marvel-single)

## 2.  Uploaded data to Roboflow: 
Upload labeled data to Roboflow .

## 3. Preprocess your data:  
This step is essential to improve the performance of your model and prevent overfitting.

## 4. Configured  YOLOv4 model:
 In Roboflow, select the YOLOv4 model architecture and customize the hyperparameters, such as the learning rate, batch size, and number of epochs. You can also use transfer learning by selecting a pre-trained YOLOv4 model and fine-tuning it on dataset.

## 5. Train model:     
Started training your YOLOv4 model using Roboflow's built-in training pipeline. The pipeline automatically sets up the data pipeline, trains the model, and saves the checkpoints.

## 6. Deployed model: 
deployed  it to my application 

For more details, please refer to the documentation article [Getting Started with YOLO v4.](https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208)








## Deployment

To deploy this project run

```bash
  git clone https://github.com/AishwaryaSatpute1/Internship_Project
  cd Internship_Project
```

```bash
  python kalman_filter+Yolo.py
```


