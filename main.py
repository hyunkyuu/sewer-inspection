from ultralytics import YOLO
import models
import pandas as pd
import numpy as np

def main():

    # models.trainer(train_model='yolov8n.pt', save_name='test02')
    models.predictor(model_path="./runs/detect/test02/weights/best.pt", 
                     source_path="./videos/raw/JCW13.mp4", 
                     live=True, 
                     save_path=True)

main()