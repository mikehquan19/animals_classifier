# TODO: Get in the theoretical details of Depthwise Convolution and SE blocks
from torch import nn, Tensor
from torchsummary import summary
from typing import Any
from configs import *


class SEBlock(nn.Module): 
    """
    Architecture of squeeze and activation block (SE)
    - Reading for [Squeeze and Excitation (SE)](https://towardsdatascience.com/introduction-to-squeeze-excitation-networks-f22ce3a43348/)
    - Research paper of [SE](https://arxiv.org/pdf/1709.01507)
    ```
    expanded_features = num_features * scale
    reduced_features = num_features // reduction
    ```
    Bottleneck from ```expanded_features``` to ```reduced_features```
    """

    def __init__(self, num_features: int, scale: int, reduction: int) -> None: 
        super().__init__()

        # Layers for squeeze and excitation 
        self.squeeze = nn.AdaptiveAvgPool2d(1) # Essentially, global avg pool 
        # safeguard against out channels ~ 0 
        self.excitation = nn.Sequential(
            nn.Conv2d(num_features * scale, max(1, num_features // reduction), 1), 
            nn.SiLU(),
            nn.Conv2d(max(1, num_features // reduction), num_features * scale, 1),
            nn.Sigmoid()
        )

    def forward(self, x) -> Tensor:
        # squeeze, excitation, and matmul 
        output = self.excitation(self.squeeze(x)) # (B, C, H, W) => (B, C, 1, 1)
        output = x * output # (B, C, H, W) * (B, C, 1, 1) => (B, C, H, W) 
        return output


class MBConvBlock(nn.Module): 
    """
    Architecture of the mobile bottleneck convolutional block 
    - Image of the architecture of [MBConv](https://www.mdpi.com/computers/computers-13-00109/article_deploy/html/images/computers-13-00109-g002-550.jpg) (super convenient and illustrious)

    Args: 
        scale (int): scaling the input channels over the layers in between (either 1 or 6)
        in_channels (int): The features of the image tensor before feeding it through the model
        out_channels (int): The feature after the model
        kernel_size (int): The size of the kernel for the depthwise convolutional block (either 3 or 5)
        stride (int): The stride for the depthwise convolutional block (either 1 or 2)
    """
    def __init__(
        self, scale: int, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None: 
        super().__init__()
        if scale not in [1, 6]: 
            raise ValueError("Illegal scale, must be either 1 or 6.")
        if kernel_size not in [3, 5]: 
            raise ValueError("Illegal kernel_size, must be either 3 or 5.")
        if not 1 <= stride <= 2: 
            raise ValueError("Illegal stride, must be either 1 or 2.")

        self.skip_connect = out_channels == in_channels and stride == 1

        # Pointwise convolution block to expand the channels (only for MBConv6)
        self.pointwise_conv_with_activation= None
        if scale == 6: 
            self.pointwise_conv_with_activation = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * scale, 1, bias=False),
                nn.BatchNorm2d(in_channels * scale), 
                nn.SiLU()) # Pytorch's swish activation

        # Depthwise convolution block
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels * scale, in_channels * scale, 
                kernel_size=kernel_size, padding=kernel_size // 2, stride=stride, 
                groups=in_channels * scale, bias=False
            ),
            nn.BatchNorm2d(in_channels * scale), 
            nn.SiLU())
 
        # SE block, reduction rate generally used for EfficientNet family = 4
        self.se = SEBlock(in_channels, scale, 4)

        # Last pointwise convolution 
        self.pointwise_conv_without_activation = nn.Sequential(
            nn.Conv2d(in_channels * scale, out_channels, 1, bias=False), 
            nn.BatchNorm2d(out_channels))

    def forward(self, x) -> Tensor: 
        if self.pointwise_conv_with_activation: 
            output = self.depthwise_conv(self.pointwise_conv_with_activation(x))
        else: 
            output = self.depthwise_conv(x)

        output = self.se(output)

        output = self.pointwise_conv_without_activation(output)
        if self.skip_connect: output = output + x
        return output


class EfficientNet(nn.Module): 
    """ 
    [EfficientNet](https://arxiv.org/pdf/1905.11946)'s architecture, the pinnacle of image classification architectures (sort of).

    Args: 
        configs (Any): The configuration for the architecture of the model 
            For example, 
            ```
            {
                "conv3": { # the conf for the first convolution
                    "out_channels": 32
                },  
                "mbconv_blocks": [ # conf for the list of set of mbconv blocks 
                    {"out_channels": 16, "kernel_size": 3, "stride": 1, "num_blocks": 1},
                    ..., 
                    {"out_channels": 100, "kernel_size": 1, "stride": 1, "num_blocks": 3},
                ],
                "conv1": { # the conf for the last convolution
                    "out_channels": 1280
                } 
            }
            ```
    """

    def __init__(self, configs: Any) -> None: 
        super().__init__()
        self.configs = configs
        # Non-properties, the initial input and output channels of the model 
        input_channels, output_channels = 3, self.configs["conv3"]["out_channels"]

        # Stage 1, the conv3x3
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU())

        # Stage 2 -> 8, each is a set of MB convolution blocks
        for ix1, config in enumerate(self.configs["mbconv_blocks"]): 
            mbConvBlocks = []
            for ix2 in range(config["num_blocks"]):
                input_channels = output_channels 
                # The output channels change for first block in every set of blocks 
                if ix2 == 0: output_channels = config["out_channels"] 

                mbConvBlocks.append(MBConvBlock(
                    scale=1 if ix1 == 0 else 6, # first set is MBConv1, other sets are MBConv6
                    in_channels=input_channels, out_channels=output_channels, 
                    kernel_size=config["kernel_size"],
                    stride=config["stride"] if ix2 == 0 else 1, # second to last block's stride = 1
                ))

            setattr(self, f"mbconv{ix1}_blocks", nn.Sequential(*mbConvBlocks))

        # Stage 9, the conv1x1 with avg-pooling and fc
        input_channels, output_channels = output_channels, self.configs["conv1"]["out_channels"]

        self.pre_fc_block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels), 
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(output_channels, 10)

    def forward(self, x) -> Tensor: 
        output = self.conv1(x)
        for ix in range(len(self.configs["mbconv_blocks"])):
            output = getattr(self, f"mbconv{ix}_blocks")(output)

        output = self.pre_fc_block(output)
        output = output.view(output.size(0), -1) # (B, C, 1, 1) => (B, C)
        return self.fc(output)