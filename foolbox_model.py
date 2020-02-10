from typing import Tuple, Union, Callable, Dict
from typing_extensions import Literal
import foolbox as fbn
from tensorflow.keras import applications as app


Mode = Union[Literal["tf"], Literal["torch"], Literal["caffe"]]


def get_bounds_and_preprocessing(
    mode: Mode,
) -> Tuple[fbn.types.BoundsInput, fbn.types.Preprocessing]:
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py

    if mode == "tf":
        return (-1, 1), None
    elif mode == "torch":
        preprocessing = dict(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3
        )
        return (0, 1), preprocessing
    elif mode == "caffe":
        mean = [103.939, 116.779, 123.68]
        preprocessing = dict(flip_axis=-1, mean=mean)  # RGB to BGR
        return (0, 255), preprocessing
    else:
        raise ValueError(f"unknown mode '{mode}'")


# https://github.com/keras-team/keras-applications/tree/master/keras_applications
Models: Dict[str, Tuple[Callable, Mode]] = {
    "densenet121": (app.densenet.DenseNet121, "torch"),
    "densenet169": (app.densenet.DenseNet169, "torch"),
    "densenet201": (app.densenet.DenseNet201, "torch"),
    # "EfficientNetB0": (app.efficientnet.EfficientNetB0, "torch"),
    # "EfficientNetB1": (app.efficientnet.EfficientNetB1, "torch"),
    # "EfficientNetB2": (app.efficientnet.EfficientNetB2, "torch"),
    # "EfficientNetB3": (app.efficientnet.EfficientNetB3, "torch"),
    # "EfficientNetB4": (app.efficientnet.EfficientNetB4, "torch"),
    # "EfficientNetB5": (app.efficientnet.EfficientNetB5, "torch"),
    # "EfficientNetB6": (app.efficientnet.EfficientNetB6, "torch"),
    # "EfficientNetB7": (app.efficientnet.EfficientNetB7, "torch"),
    "InceptionResNetV2": (app.inception_resnet_v2.InceptionResNetV2, "tf"),
    "InceptionV3": (app.inception_v3.InceptionV3, "tf"),
    "MobileNet": (app.mobilenet.MobileNet, "tf"),
    "MobileNetV2": (app.mobilenet_v2.MobileNetV2, "tf"),
    "NASNetLarge": (app.nasnet.NASNetLarge, "tf"),
    "NASNetMobile": (app.nasnet.NASNetMobile, "tf"),
    "ResNet50": (app.resnet.ResNet50, "caffe"),
    "ResNet101": (app.resnet.ResNet101, "caffe"),
    "ResNet152": (app.resnet.ResNet152, "caffe"),
    # "ResNet50": (app.resnet50.ResNet50, "caffe"),
    "ResNet50V2": (app.resnet_v2.ResNet50V2, "tf"),
    "ResNet101V2": (app.resnet_v2.ResNet101V2, "tf"),
    "ResNet152V2": (app.resnet_v2.ResNet152V2, "tf"),
    "ResNeXt50": (app.resnext.ResNeXt50, "torch"),
    "ResNeXt101": (app.resnext.ResNeXt101, "torch"),
    "VGG16": (app.vgg16.VGG16, "caffe"),
    "VGG19": (app.vgg19.VGG19, "caffe"),
    "Xception": (app.xception.Xception, "tf"),
}


def create(name: str):
    Model, mode = Models[name]
    model = Model(weights="imagenet")
    bounds, preprocessing = get_bounds_and_preprocessing(mode)
    fmodel = fbn.TensorFlowModel(model, bounds=bounds, preprocessing=preprocessing)
    return fmodel
