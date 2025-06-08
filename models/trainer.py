from ultralytics import YOLO

def trainer(train_model=None, yaml_path="assets/configs.yaml", epochs=100, save_name=None):
    """
    Trains the model.

    :param train_model: the type of the model to train.
    :param yaml_path: the path to the yaml file.
    :param epochs: the number of epochs.
    :param save_name: the name of the saved model.

    :type model_path: str, default None
    :type source_path: str, default 0
    :type epochs: int, default None
    :type save_name: str, default None

    :return: None
    :rtype: None    
    """
    
    model = YOLO(train_model)
    model.train(
        data=yaml_path,
        epochs=epochs,
        name=save_name
    )
