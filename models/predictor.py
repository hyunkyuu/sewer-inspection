from ultralytics import YOLO

def predictor(model_path=None, source_path=0, show=True, save=True):

    model = YOLO(model_path)
    model(
        source=source_path, 
        show=True, 
        save=True
    )
