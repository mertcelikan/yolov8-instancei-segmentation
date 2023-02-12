import os
HOME = os.getcwd()
print(HOME)
#!pip install ultralytics

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

from IPython.display import display, Image

#!mkdir {HOME}/datasets
#%cd {HOME}/datasets

#!pip install roboflow --quiet


from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("new-workspace-crr3c").project("r2p2")
dataset = project.version(2).download("yolov5")

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.yaml")  # build a new model from scratch
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
model.train(data="coco128-seg.yaml", epochs=100, imgsz=640)

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmenation masks outputs
    probs = result.probs  # Class probabilities for classification outputs