# THE MODEL
from torch import nn, Tensor
from torch.nn import functional as F

class SmallerResBlock(nn.Module): 
    """ 
    Architecture of the residual block with 2 layers used for ```SmallerResNet```

    The resblocks (```in_channels -> out_channels```) consists of: 
        - Convolution layer: ```in_channels -> out_channels```, 
        - Convolution layer: ```out_channels -> out_channels```, 

    For example, 
        - Resblock 64 -> 128, ```in_channels = 64``` & ```out_channels = 128```
        - Resblock 512 -> 512, ```in_channels = 512``` & ```out_channels = 512```

    Args: 
        in_channels (int): Channels of the image tensor before feeding it through the resblock.
        out_channels (int): Channels of image tensor after feeding it through the resblock.
        stride (int): Stride for first convolution layer of the block
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super(SmallerResBlock, self).__init__()

        if stride != 1 and stride != 2: 
            raise ValueError("Number of residual block's strides should be 1 or 2")
        
        # Pre-skip
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        # Batch normalization 
        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

        # Downsample applied when tensors' dimensions don't match 
        self.downsample = None 
        if in_channels != out_channels: 
            # Downsample includes both convolution and batch norm 
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(num_features=out_channels))

    def forward(self, x) -> Tensor: 
        """ Return """
        output = F.relu(self.batch_norm1(self.conv1(x)))
        output = self.batch_norm2(self.conv2(output))

        # skip connection 
        # downsample if shape of tensors in the skip connections don't match 
        residual = self.downsample(x) if self.downsample else x 
        # apply dropout after skip connection 
        output = F.dropout2d(F.relu(output + residual), p=0.2)
        return output 


class SmallerResNet(nn.Module): 
    """
    Resnet-based generic architecture of the model classifying animals' images using ```SmallerResBlock```.
    Only apply from ```Resnet-18 & Resnet-34```

    Architecture summary: 
        - conv1 layer: 7x7 convolutional layer with stride 2
        - conv2 blocks: 3x3 max-pool with stride 2, smaller resblocks 64 -> 64
        - conv3 blocks: smaller resblocks from 64 -> 128 & 128 -> 128
        - conv4 blocks: smaller resblocks from 128 -> 256 & 256 -> 256
        - conv5 blocks: smaller resblocks from 256 -> 512 & 512 -> 512
    """
    def __init__(self, num_blocks_list: list[int]) -> None:
        super(SmallerResNet, self).__init__()

        if len(num_blocks_list) != 4: 
            raise ValueError("Invalid number of residual blocks!")

        # Non-properties, used for constructing blocks, initial number of input and output channels
        input_channels, output_channels = 64, 64

        # conv1 laye r
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=input_channels), 
            nn.ReLU())

        # conv2 blocks to conv5 blocks 
        for ix1, num_blocks in enumerate(num_blocks_list): 
            conv_blocks = [] 
            # max pool before the first set of residual blocks
            if ix1 == 0: conv_blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

            for ix2 in range(num_blocks): 
                # for every set except the first, first block's stride is 2
                stride = 2 if ix2 == 0 and ix1 != 0 else 1
                # input channels starting from the second block to final block
                if ix2 == 1: input_channels = output_channels
                conv_blocks.append(SmallerResBlock(input_channels, output_channels, stride))

            # initialize the set of residual blocks 
            setattr(self, f"conv{ix1 + 2}_blocks", nn.Sequential(*conv_blocks))
            # output channels for the next set of blocks 
            if ix1 != len(num_blocks_list) - 1: output_channels *= 2 
        
        # fully-connected layer 
        self.fc = nn.Linear(output_channels, 10) # 10 categories of animal


    def forward(self, x) -> Tensor: 
        output = F.dropout2d(self.conv1(x), p=0.2)
        for ix in range(4): 
            output = getattr(self, f"conv{ix + 2}_blocks")(output)

        output = F.avg_pool2d(output, kernel_size=7)
        # Flatten the image before feeding them through the linear layer
        output = output.view(output.size(0), -1)
        return self.fc(output)
    

class LargerResBlock(nn.Module):
    """
    Architecture of the residual block with 3 layers used for ```LargerResNet```. 
    The first residual block in ResNet has stride of 2. 

    The resblock (```in_channels -> out_channels```) consists of: 
        - Convolution layer: ```in_channels -> out_channels / 4```, 
        - Convolution layer: ```out_channels / 4 -> out_channels / 4```, 
        - Convolution layer: ```out_channels / 4 -> out_channels```

    For example, 
        - Resblock 64 -> 256, ```in_channels = 64``` & ```out_channels = 256```
        - Resblock 512 -> 1024, ```in_channels = 512``` & ```out_channels = 1024```
        - Resblock 512 -> 512, ```in_channels = 512``` & ```out_channels = 512```

    Args: 
        in_channels (int): Channels of the image tensor before feeding it through the resblock.
        out_channels (int): Channels of image tensor after feeding it through the resblock.
        stride (int): Stride for first convolution layer of the block
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int=1) -> None:
        super(LargerResBlock, self).__init__()

        if not 1 <= stride <= 2:
            raise ValueError("Number of residual block's strides should be 1 or 2")
        if out_channels % 4 != 0: 
            raise ValueError("Illegal output channels, must be divisible to 4")
        
        # pre skip-connection convolutional block 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=stride, bias=False), 
            nn.BatchNorm2d(num_features=out_channels // 4),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels // 4), 
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels))

        # Downsample 
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), 
            nn.BatchNorm2d(num_features=out_channels)) if in_channels != out_channels else None

    def forward(self, x) -> Tensor:
        output = self.conv3(self.conv2(self.conv1(x)))

        # residual skip connection
        residual = self.downsample(x) if self.downsample else x # apply downsample in case of mismatch
        return F.dropout2d(F.relu(output + residual), p=0.2)


class LargerResNet(nn.Module):
    """
    Resnet-based generic architecture of the model classifying animals' images using ```LargerResBlock```.
    Only apply for ```Resnet50, Resnet101, & Resnet152``` 

    Architecture summary: 
        - Conv1 layer: 7x7 convolutional layer 3 -> 64 with stride 2
        - Conv2 blocks: 3x3 max-pool with stride 2, larger resblocks 64 -> 256 & 256 -> 256
        - Conv3 blocks: larger resblocks from 256 -> 512 & 512 -> 512
        - Conv4 blocks: larger resblocks from 512 -> 1024 & 1024 -> 1024
        - Conv5 blocks: larger resblocks from 1024 -> 2048 & 2048 -> 2048

    Args: 
        num_blocks_list (int): Number of blocks in each set of blocks
    """
    
    def __init__(self, num_blocks_list: list[int]) -> None:
        super(LargerResNet, self).__init__()

        if len(num_blocks_list) != 4: 
            raise ValueError("Illegal number of set of residual blocks!")
        
        # Non-properties used for building blocks, 
        # Initial input channels is 64 and output channels is 256 for the set of residual blocks
        input_channels, output_channels = 64, 256

        # conv1 layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=input_channels), 
            nn.ReLU())

        # conv2 - conv5 blocks 
        for ix1, num_blocks in enumerate(num_blocks_list):
            conv_blocks = []
            # max pool for the first set of residual blocks
            if ix1 == 0: conv_blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

            for ix2 in range(num_blocks):
                # every set except the first, first block's stride is 2 
                this_stride = 2 if ix1 != 0 and ix2 == 0 else 1
                # input channels starting from the second block to final block
                if ix2 == 1: input_channels = output_channels
                # add block to the set 
                conv_blocks.append(LargerResBlock(input_channels, output_channels, this_stride))

            setattr(self, f"conv{ix1 + 2}_blocks", nn.Sequential(*conv_blocks))
            # the number of output channels for the next set of resblocks until last one. 
            if ix1 != len(num_blocks_list) - 1: output_channels *= 2

        # 1 fully-connected layer
        self.fc = nn.Linear(output_channels, 10)

    def forward(self, x) -> Tensor:
        output = F.dropout2d(self.conv1(x), p=0.2) 
        for i in range(4): 
            output = getattr(self, f"conv{i + 2}_blocks")(output)
            
        output = output.view(F.avg_pool2d(output, kernel_size=7).size(0), -1)
        return self.fc(output)
    