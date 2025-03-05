# Breast-Cancer-Detector
Hackathon project

## Inspiration
In my biology lessons, I learned about the causes of cancer and I also learned that the highest cause of death in women is breast cancer. One of the reasons is late diagnosis and misdiagnosis. So I thought of creating a web app that detects breast cancer early with the help of computer vision.

## What it does
Detects breast cancer from x-ray images with instance segmantation.

## How we built it
For my project i used vscode as editor and trained my YOLOv8 model with a x-ray images dataset, the [notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-instance-segmentation-on-custom-dataset.ipynb#scrollTo=D2YkphuiaE7_) from Roboflow make it possible. The libraries i used and for what are:  

Dash --> framework, create web app
plotly --> data visualisation library, create figures and show images



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
