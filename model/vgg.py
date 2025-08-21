# NOTE: The number of params between these layers don't differ significantly
#       therefore, VGG with more layers don't neccesarily perform better.
#       And training VGG networks will also be painfully long because of large number of params

from torch import nn, Tensor
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

    def __init__(self, in_channels: int, out_channels: int, num_layers: int) -> None:
        super(ConvBlock, self).__init__()

        if num_layers > 4:
            raise ValueError("The convolutional block should have at max 4 conv layers")
        
        self.num_layers = num_layers
        # Convolutional layer along with batch normalization and Relu activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU())

        for ix in range(self.num_layers - 1): 
            setattr(self, f"conv{ix + 2}", nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels), 
                nn.ReLU()))

    def forward(self, x) -> Tensor:
        # Feed through the number of available layers of convolution, dropout, and maxpool
        output = self.conv1(x)
        for ix in range(self.num_layers - 1): 
            output = getattr(self, f"conv{ix + 2}")(output)
        # Apply the dropout, max pool
        return F.max_pool2d(F.dropout(output, p=0.2), kernel_size=2)


class GenericVGG(nn.Module):
    """
    VGG-Based generic architecture of the model classifying animals' images using ```ConvBlock```.

    Architecture summary: 
        - Conv1 block 3 -> 64
        - Conv2 block 64 -> 128
        - Conv3 block 128 -> 256
        - Conv4 block 256 -> 512
        - Conv5 block 512 -> 512

    Args: 
        num_layers_list (int): The list of number of layers of each convolutional block
    """
    
    def __init__(self, num_layers_list: list[int]) -> None:
        super(GenericVGG, self).__init__()
        
        if len(num_layers_list) != 5: 
            raise ValueError("Illegal number of convolutional blocks")
        
        # Non-properties. 
        # The initial number of input and output channels for the model architecture
        input_channels, output_channels = 3, 64

        for ix, num_layers in enumerate(num_layers_list): 
            setattr(self, f"conv{ix+1}_block", ConvBlock(input_channels, output_channels, num_layers))
            input_channels = output_channels
            # Output channels doubles until second to last convolutional block
            if ix < len(num_layers_list) - 2: output_channels *= 2

        # Fully connected layers (3 layers)
        self.fc1 = nn.Linear(output_channels * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10) # 10 categories of animals

    def forward(self, x) -> Tensor:
        # Feed the image input through all 5 convolutional blocks. 
        for i in range(5): 
            if i == 0:  
                output = getattr(self, f"conv{i + 1}_block")(x)
            else: 
                output = getattr(self, f"conv{i + 1}_block")(output)

        # Then Ffatten the image and then 3 fully connected layers
        output = output.view(output.size(0), -1)
        output = F.dropout2d(F.relu(self.fc1(output)), p=0.2)
        output = F.dropout2d(F.relu(self.fc2(output)), p=0.2) 
        return self.fc3(output)