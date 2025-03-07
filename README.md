# Breast-Cancer-Detector
Hackathon project

## Inspiration
In my biology lessons, I learned about the causes of cancer and I also learned that the highest cause of death in women is breast cancer. One of the reasons are late diagnosis and misdiagnosis. So I thought of creating a web app that detects breast cancer early with the help of object detection.

## What it does
Detects breast cancer from x-ray images with instance segmantation.

## How we built it
For my project i used vscode as editor and trained my YOLOv8 model with a x-ray images dataset, the [notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-instance-segmentation-on-custom-dataset.ipynb#scrollTo=D2YkphuiaE7_) from Roboflow make it possible. The libraries I used and for what are:  

dash --> web framework; create web app  
plotly --> data visualisation library; create figures and show images  
random --> library for generate random number; create random colors for each class 
io --> module for dealing various types of input/output;  I/O operations on byte data    
base64 --> module for encoding and decoding binary data; decode binary data
opencv --> computer vision library for image processing; resize, convert color(BGR to RGB), draw boxes and filled polygons and overlay segmentation masks
numpy --> numerical computing library; convert mask as array and reshape arrays  
ultralytics --> deep learning object detection framework; load YOLOv8 model for instance segmentation

## Challenges I ran into
Problem: Not enough computing power to train my YOLO model with my customer dataset  
Solution: reduces epochs  
Consequence: reduces accuracy  

## Accomplishments that we're proud of
That my project even work.

## What I learned
I learned how instance segmentation works and the dangers of ignorance and late diagnosis of breast cancer.

## What's next for Breast Cancer Detection
Train my YOLO model more so it becomes better and train it with other cancer types datasets.

## How it works
You just upload a x-ray image and in few seconds the program finds out whether it is breast cancer or not.

## Video
You can see the loom video [here](https://www.loom.com/share/f5338a3ee4a34e0084b9089c0a2ab723?sid=cb066360-7718-4667-af75-2c6da78560b6)
