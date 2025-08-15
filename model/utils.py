import torch
from torch import nn
from .vgg import *
from .resnet import *
from torchsummary import summary

def get_model(name: str, checkpoint: str | None = None) -> nn.Module: 
    """ 
    Return the model based on name and load the weights if given the checkpoint

    Args: 
        name (str): The name of the model
        checkpoint (str | None): The name of the file from which to load the weights of the model

    """

    # For VGG models 
    if name == "vgg11": 
        model = VGG11Classifier()
    elif name == 'vgg13':
        model = VGG13Classifier()
    elif name == 'vgg16':
        model = VGG16Classifier()
    elif name == 'vgg19':
        model = VGG19Classifier()

    # For ResNet models 
    elif name == 'resnet18':
        model = ResNet18Classifier()
    elif name == 'resnet34':
        model = ResNet34Classifier()
    elif name == 'resnet50':
        model = ResNet50Classifier()
    elif name == 'resnet101':
        model = ResNet101Classifier()
    elif name == 'resnet152':
        model = ResNet152Classifier()
    else: 
        raise Exception("Invalid model's name! Model is either Resnet or VGG and the number of layers must be valid")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device=device)
    if checkpoint: 
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    return model


if __name__ == "__main__": 
    """ This is to check the architecture of the model """ 
    summary(model=get_model("resnet50"), input_size=(3, 224, 224))