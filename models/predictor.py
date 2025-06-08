from ultralytics import YOLO

def predictor(model_path=None, source_path=0, live=False, save_path=None):
    """
    Performs prediction on the given source using the provided model.

    :param model_path: the path of the model.
    :param source_path: the path of the source. if `source=0`, use webcam
    :param show: choose to show the result live or not.
    :param save: the path to save the result.

    :type model_path: str, default None
    :type source_path: int or str, default 0
    :type show: bool, default True
    :type save: str, default None

    :return: None
    :rtype: None
    """

    if model_path is None:
        raise ValueError("Can't find path to model.")
    if source_path is None:
        raise ValueError("Can't find path to source.")
    if save_path is None:
        raise ValueError("Can't find path to save.")

    model = YOLO(model_path)
    model(
        source=source_path, 
        show=live, 
        save=save_path
    )
