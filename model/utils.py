import torch
from torch import nn
from vgg import *
from resnet import *
from efficientnet import *
from configs import *
from torchsummary import summary

def get_model(name: str, checkpoint: str | None = None) -> nn.Module: 
    """ 
    Return the model using name and load the pretrained weights if given the checkpoint

    Args: 
        name (str): The name of the model
        checkpoint (str | None): The name of the file from which to load the pretrained weights
    """

    # For VGG models 
    if name == "vgg11": 
        model = GenericVGG(vgg11_num_layers_list)
    elif name == 'vgg13':
        model = GenericVGG(vgg13_num_layers_list)
    elif name == 'vgg16':
        model = GenericVGG(vgg16_num_layers_list)
    elif name == 'vgg19':
        model = GenericVGG(vgg19_num_layers_list)

    # For ResNet models 
    if name == 'resnet18':
        model = SmallerResNet(resnet18_num_blocks_list)
    elif name == 'resnet34':
        model = SmallerResNet(resnet34_num_blocks_list)
    elif name == 'resnet50':
        model = LargerResNet(resnet50_num_blocks_list)
    elif name == 'resnet101':
        model = LargerResNet(resnet101_num_blocks_list)
    elif name == 'resnet152':
        model = LargerResNet(resnet152_num_blocks_list)

    # For EfficientNet models 
    if name == 'efficientnetb0':
        model = EfficientNet(efficientnet_b0_conf)
    elif name == 'efficientnetb1':
        model = EfficientNet(efficientnet_b1_conf)
    elif name == 'efficientnetb2': 
        model = EfficientNet(efficientnet_b2_conf)
    elif name == 'efficientnetb3': 
        model = EfficientNet(efficientnet_b3_conf)
    else: 
        raise Exception("Invalid model's name! Model is either ResNet, VGG, or EfficientNet and the number of layers must be valid")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device=device)
    if checkpoint: 
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    return model


if __name__ == "__main__": 
    """ This is to check the architecture of the model """ 
    summary(model=get_model("vgg19"), input_size=(3, 224, 224))