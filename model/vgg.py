# THE MODELs
import torch
from torch import nn, cuda
from torch.nn import functional as F
from torchsummary import summary

# TODO: Create the generic VGG that can be applied to both VGG-16 and VGG-19

class ConvBlock(nn.Module):
    """
    Architecture of the convolutional block.
    Default block has 2 layers of convolution and max pool. Block can have 3 layers
    """

    def __init__(self, in_channels, out_channels, num_layers=2):
        super(ConvBlock, self).__init__()

        if num_layers != 2 and num_layers != 3:
            raise Exception("The convolutional block should have only 2 or 3 conv layers")
        self.num_layers = num_layers

        # Convolutional layer along with batch normalization 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels), 
            nn.ReLU())
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU())

    def forward(self, x):
        # Feed through 2 layers of convolution, batchnorm, and dropout
        output = self.conv2(self.conv1(x))
        # apply one more layer if it's a 3-layer block
        if self.num_layers == 3: output = self.conv3(output)
        # apply the dropout, max pool and return
        return F.max_pool2d(F.dropout(output, p=0.2), kernel_size=2)


class VGG16Classifier(nn.Module):
    """
    Architecture of the model classifying animals' images,
    based on VGG-16 (possibly 19 later)
    """

    def __init__(self):
        super(VGG16Classifier, self).__init__();

        # The initial number of channels of the image is tranformed from 3 to 
        num_channels = 64

        # Two 2-layer blocks (4 layers)
        two_layer_blocks = [ConvBlock(3, num_channels), ConvBlock(num_channels, num_channels * 2)]
        num_channels *= 2 # 128
        self.two_layer_blocks = nn.Sequential(*two_layer_blocks)

        # Three 3-layer blocks (9 layers)
        three_layer_blocks = []
        for _ in range(2):
            three_layer_blocks.append(ConvBlock(num_channels, num_channels * 2, 3))
            num_channels *= 2 # 256, 512
        three_layer_blocks.append(ConvBlock(num_channels, num_channels, 3))
        self.three_layer_blocks = nn.Sequential(*three_layer_blocks)

        # Fully connected layers (3 layers)
        self.fc1 = nn.Linear(num_channels * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10) # 10 categories of animals

    def forward(self, x):
        """
        Feed them the image input through all 5 blocks (13 layers of conv)
        Flatten the image and then 3 fully connected layers
        """

        output = self.two_layer_blocks(x)
        output = self.three_layer_blocks(output)

        # Flatten the image
        output = output.view(output.size(0), -1)

        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        return self.fc3(output)


def main(): 
    # The model
    animal_classifier = VGG16Classifier().to(
        torch.device("cuda") if cuda.is_available() else torch.device("cpu"))
    print(summary(animal_classifier, input_size=(3, 224, 224)))