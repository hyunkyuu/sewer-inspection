from ultralytics import YOLO

def trainer(train_model, yaml_path, save_path):
    
    model = YOLO(train_model)
    model.train(
        data=yaml_path,     # yaml path
        epochs=100,         # num of trains
        name=save_path      # save path
    )
