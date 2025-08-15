# NOTE: The number of params between these layers don't differ significantly
#       therefore, VGG with more layers don't neccesarily perform better.
#       And training VGG networks will also be painfully long because of large number of params

from torch import nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    """
    Architecture of the convolutional block used for ```GenericVGG```. Block can have up to 4 layers

    The convblocks (```in_channels -> out_channels```) with upt to 4 layers consists of: 
        - Convolution layer: ```in_channels -> out_channels```, 
        - Convolution layer: ```out_channels -> out_channels``` (optional), 
        - Convolution layer: ```out_channels -> out_channels``` (optional), 
        - Convolution layer: ```out_channels -> out_channels``` (optional), 
        
    Args: 
        in_channels (int): The number of channels of image tensor before feeding them
        out_channels (int): The number of channels of image tensor after feeding them
        num_layers (int): The number of convolutional layers of a block
        
    """

    def __init__(self, in_channels: int, out_channels: int, num_layers: int):
        super(ConvBlock, self).__init__()
        if num_layers > 4:
            raise Exception("The convolutional block should have at max 4 conv layers")
        
        self.num_layers = num_layers
        # Convolutional layer along with batch normalization and Relu activation
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU())

        for ix in range(self.num_layers - 1): 
            setattr(self, f"conv_block{ix + 2}", nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels), 
                nn.ReLU()))

    def forward(self, x):
        # Feed through the number of available layers of convolution, dropout, and maxpool
        output = self.conv_block1(x)
        if self.num_layers >= 2: output = self.conv_block2(output)
        if self.num_layers >= 3: output = self.conv_block3(output)
        if self.num_layers == 4: output = self.conv_block4(output)
        # apply the dropout, max pool and return the tensor
        return F.max_pool2d(F.dropout(output, p=0.2), kernel_size=2)


class GenericVGG(nn.Module):
    """
    VGG-Based generic architecture of the model classifying animals' images 
    using ```ConvBlock```.

    Args: 
        num_layers_list (int): The list of number of layers of each convolutional block
    """

    def __init__(self, num_layers_list):
        super(GenericVGG, self).__init__()
        if len(num_layers_list) != 5: 
            raise Exception("Invalid number of convolutional blocks")
        # Non-properties. The initial number of channels to which the image is tranformed from 3 
        input_channels, output_channels = 3, 64

        # conv blocks 3 -> 64
        # conv2 block 64 -> 128
        # conv3 block 128 -> 256
        # conv4 block 256 -> 512
        # conv5 block 512 -> 512
        for ix, num_layers in enumerate(num_layers_list): 
            setattr(self, f"conv{ix+1}_block", ConvBlock(input_channels, output_channels, num_layers))
            input_channels = output_channels
            if ix < len(num_layers_list) - 2: output_channels *= 2

        # Fully connected layers (3 layers)
        self.fc1 = nn.Linear(output_channels * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10) # 10 categories of animals

    def forward(self, x):
        """ 
        Feed them the image input through all 5 convolutional blocks. 
        Flatten the image and then 3 fully connected layers 
        """
        output = self.conv1_block(x)
        output = self.conv5_block(self.conv4_block(self.conv3_block(self.conv2_block(output))))

        output = output.view(output.size(0), -1)
        output = F.relu(self.fc2(F.relu(self.fc1(output))))
        return self.fc3(output)
    

class VGG11Classifier(GenericVGG): 
    """ 11-layer VGG """
    def __init__(self):
       super().__init__(num_layers_list=[1, 1, 2, 2, 2])


class VGG13Classifier(GenericVGG): 
    """ 13-layer VGG """
    def __init__(self):
       super().__init__(num_layers_list=[2, 2, 2, 2, 2])


class VGG16Classifier(GenericVGG): 
    """ 16-layer VGG """
    def __init__(self): 
        super().__init__(num_layers_list=[2, 2, 3, 3, 3])


class VGG19Classifier(GenericVGG): 
    """ 19-layer VGG """
    def __init__(self):
       super().__init__(num_layers_list=[2, 2, 4, 4, 4])