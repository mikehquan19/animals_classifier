# THE MODEL
import torch
from torch import nn, cuda
from torch.nn import functional as F
from torchsummary import summary

class SmallerResBlock(nn.Module): 
    """ Architecture of the residual block used for smaller ResNet """
    
    def __init__(self, in_channels: int, out_channels: int, stride):
        super(SmallerResBlock, self).__init__()
        if stride != 1 and stride != 2: 
            raise Exception("Number of resblock's strides should be 1 or 2")
        
        # Pre-skip
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False)

        # Batch normalization 
        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

        # Downsample applied when tensors' dimensions don't match 
        self.downsample = None 
        if in_channels != out_channels: 
            # Downsample includes both convolution and batch norm 
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), 
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x): 
        """ Return  """
        output = F.relu(self.batch_norm1(self.conv1(x)))

        # skip connection 
        output = self.batch_norm2(self.conv2(output))
        # downsample if shape of tensors in the skip connections don't match 
        residual = self.downsample(x) if self.downsample else x 
        # apply dropout after skip connection 
        output = F.dropout2d(F.relu(output + residual), p=0.2)
        return output 


class SmallerResNet(nn.Module): 
    """
    Generic architecture of the model classifying animals' images, based on ResNet 
    Only apply from 18 -> 34 layers 
    """
    def __init__(self, num_blocks_list):
        super(SmallerResNet, self).__init__()

        input_channels = 64 
        output_channels = 64 

        # conv1 layer: 7x7 convolutional layer with stride 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, output_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=output_channels), 
            nn.ReLU(),
        )

        for i1, num_blocks in enumerate(num_blocks_list): 
            conv_blocks = [] 

            # max pool for the first set of residual blocks
            if i1 == 0: 
                conv_blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

            for i2 in range(num_blocks): 
                # first blocks's stride is 2 to reduce the input size by factor 2
                stride = 2 if i2 == 0 and i1 != 0 else 1
                # add block to the set of blocks 
                conv_blocks.append(SmallerResBlock(input_channels, output_channels, stride))

            # initialize the set of residual blocks 
            setattr(self, f"conv{i1 + 2}_blocks", nn.Sequential(*conv_blocks))
            output_channels *= 2 
        
        # fully-connected layer 
        self.fc = nn.Linear(output_channels * 2, 10)


    def foward(self, x): 
        output = F.dropout2d(self.conv1(x), p=0.2)
        output = self.conv2_blocks(output)
        output = self.conv3_blocks(output)
        output = self.conv4_blocks(output)
        output = self.conv5_blocks(output)

        # Apply the average pool
        output = F.avg_pool2d(output, kernel_size=7)
        # Flatten the image before feeding them through the linear layer
        output = output.view(output.size(0), -1)
        return self.fc(output)
    

class ResNet18Classifier(SmallerResBlock): 
    # 18-layer ResNet 
    def __init__(self):
        super().__init__(num_blocks_list=[2, 2, 2, 2])


class ResNet34Classifier(SmallerResBlock): 
    # 34-layer ResNet 
    def __init__(self):
        super().__init__(num_blocks_list=[3, 4, 6, 3])


class LargerResBlock(nn.Module):
    """
    Architecture of the residual block. The first residual block in ResNet has stride of 2
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(LargerResBlock, self).__init__()
        if stride != 1 and stride != 2:
            raise Exception("Number of resblock's strides should be 1 or 2")
        
        # pre skip-connection convolutional block 
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), 
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels), 
            nn.ReLU(),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels * 4),
        )

        # Downsample 
        self.downsample = None
        if in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, stride, bias=False), 
                nn.BatchNorm2d(num_features=out_channels * 4)
            )

    def forward(self, x):
        output = self.conv_block1(x)
        output = self.conv_block2(output)

        # skip connection
        output = self.conv_block3(output)
        residual = x
        # apply downsample 
        if self.downsample: residual = self.downsample(x)
        return F.dropout2d(F.relu(output + residual), p=0.2)


class LargerResNet(nn.Module):
    """
    Generic architecture of the model classifying animals' images, based on ResNet 
    Only apply from 50 -> 152 layers 
    """
    
    def __init__(self, num_blocks_list):
        super(LargerResNet, self).__init__()

        # initial input and output channels is 64
        input_channels = 64
        output_channels = 64 

        # conv1 layer: 7x7 convolutional layer with stride 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, output_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=output_channels), 
            nn.ReLU(),
        )

        """
        conv2 blocks: 3x3 max-pool with stride 2, resblocks 64 -> 256
        conv3 blocks: resblocks from 128 -> 512
        conv4 blocks: resblocks from 256 -> 1024
        conv5 blocks: reblocks from 512 -> 2048
        """

        for ix1, num_blocks in enumerate(num_blocks_list):
            conv_blocks = []
            if ix1 == 0:
                # max pool for the first set of residual blocks
                conv_blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

            for ix2 in range(num_blocks):
                # first block's stride is 2 to reduce the input size by factor of 2
                this_stride = 2 if (ix1 != 0 and ix2 == 0) else 1
                # input channels of the second block to final block
                if ix2 != 0: input_channels = output_channels * 4
                # add block to the set 
                conv_blocks.append(LargerResBlock(input_channels, output_channels, this_stride))

            # initialize the set of residual blocks
            setattr(self, f"conv{ix1 + 2}_blocks", nn.Sequential(*conv_blocks))

            # the number of channels for the next round
            output_channels *= 2

        # 1 fully-connected layer
        self.fc = nn.Linear(output_channels * 2, 10)

    def forward(self, x):
        # Feed the image to all convolutional layers
        output = F.dropout2d(self.conv1(x), p=0.2) # apply dropout 
        output = self.conv2_blocks(output)
        output = self.conv3_blocks(output)
        output = self.conv4_blocks(output)
        output = self.conv5_blocks(output)

        output = F.avg_pool2d(output, kernel_size=7)
        output = output.view(output.size(0), -1)
        return self.fc(output)
    

class ResNet50Classifier(LargerResNet): 
    # 50-layer ResNet 
    def __init__(self):
        super().__init__(num_blocks_list=[3, 4, 6, 3])


class ResNet101Classifier(LargerResNet): 
    # 101-layer ResNet 
    def __init__(self):
        super().__init__(num_blocks_list=[3, 4, 23, 3])


class ResNet152Classifier(LargerResNet): 
    # 152-layer ResNet 
    def __init__(self): 
        super().__init__(num_blocks_list=[3, 8, 36, 3])


def main(): 
    # this is to check the architecture of the model 
    summary(
        model=ResNet101Classifier().to(
            torch.device("cuda") if cuda.is_available() else torch.device("cpu")
        ), 
        input_size=(3, 224, 224)
    )